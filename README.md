# Dynamic Deep Survival Analysis for Limit Order Execution Under Adverse Selection

A research codebase that treats **passive limit order execution as a competing-risks survival problem**. Instead of asking "will this order fill?", the models estimate *when* a resting limit order fills and *which kind* of fill it is:

| Event | Code | Meaning |
|---|---|---|
| `CENSORED` | 0 | Order never filled within the observation horizon (or labeling was invalid) |
| `FAVORABLE_FILL` | 1 | Order filled and the post-trade mid-price move was benign |
| `TOXIC_FILL` | 2 | Order filled and the market moved adversely afterwards (adverse selection) |

Cause-specific cumulative incidence functions (CIFs) from a DeepHit-style model are then used as a **trading policy**: submit, hold, or cancel a passive order based on the predicted toxic-fill risk, and evaluate the resulting policy by implementation shortfall in a market-by-order replay backtest.

---

## 1. What the project actually does

1. **Streams raw MBO (market-by-order) data** from Databento (`XNAS.ITCH`) and rebuilds a full limit order book message by message.
2. **Injects virtual orders** into the reconstructed book at sampled points in time, tracks their queue position, and records their lifetime until fill or censoring.
3. **Labels each virtual order** with a competing-risks event type, using an adaptively-selected post-trade markout window.
4. **Trains competing-risks survival models** (DeepHit / Dynamic-DeepHit) over LOB feature sequences, with four interchangeable sequence backbones.
5. **Backtests the model as an execution policy**, either over the labeled dataset or over a full raw-feed replay with explicit inference-latency modeling.

---

## 2. Repository layout

```
src/
  config.py                  Single source of truth for all shared parameters (CONFIG)
  lob_implementation.py      Order book reconstruction from MBO messages (Book, Market, PriceLevel)
  order_tracking.py          Virtual-order sampling, queue tracking, parallel DBN -> Parquet build
  raw_splits.py              Chunking of raw DBN files at "empty market" boundaries
  domain/enums.py            EventType (CENSORED / FAVORABLE_FILL / TOXIC_FILL)

  features/
    base.py                  BaseLOBTransform interface
    representation.py        Mid-price-centered LOB representations
    compose.py               Toxicity/microstructure features + transform composition

  labeling/
    base.py                  BaseLabeler interface
    competing_risks.py       ExecutionCompetingRisksLabeler (favorable vs toxic vs censored)
    window_selecting.py      MarkoutAnalyzer + StabilizationWindowSelector
    utils.py                 Window-suffix helpers

  models/
    base.py                  BaseDeepHitCompetingModel: shared projection, attention pooling,
                             cause-specific heads, auxiliary head, forward pass
    gru.py                   DeepHitRNNCompeting
    gru_transformer.py       DeepHitRNNTransformerCompeting
    transformer.py           DeepHitTransformerCompeting
    mamba.py                 DeepHitMambaCompeting (optional mamba_ssm dependency)

  notebook_data.py           Dynamic-sample manifests, splits, normalization, discretization
  notebook_losses.py         Dynamic/static DeepHit losses (L1 + alpha*L2 + L3)
  notebook_evaluation.py     Brier score and uninformed baselines
  notebook_setup.py          Environment/compiler bootstrap helpers for GPU notebooks

  backtest/
    types.py                 MarketSnapshot, TradingDecision, BacktestResult, DecisionAction
    data.py                  BacktestDataset + BacktestFeatureBuilder (labeled-dataset path)
    engine.py                BacktestEngine (labeled replay, optional lifecycle-aware mode)
    raw_engine.py            RawDatabentoBacktestEngine (full MBO replay + latency)
    execution.py             RawBacktestOrder lifecycle during raw replay
    live_features.py         LiveFeatureBuilder for on-the-fly feature construction
    latency.py               StaticLatencyProvider / MeasuredLatencyProvider
    metrics.py               ImplementationShortfallMetric, toxic-cost window selection
    reports.py               BacktestReport aggregation, time-weighted shortfall
    book_utils.py            Book -> snapshot helpers, queue-ahead computation
    strategies/
      base.py                BaseStrategy / DecisionLogic contracts
      baseline.py            AlwaysPlaceLimitOrderStrategy
      deephit.py             DeepHitStrategy + threshold and toxic-CIF decision logics

scripts/                     Runnable pipeline stages (see §4)
notebooks/                   Training, grid search, EDA, ablation, and demo notebooks
model_configs/               JSONL architecture grids (parameter-budgeted)
tests/                       pytest suite (~100 tests)
references/                  Background papers and monograph (PDF)
```

---

## 3. Core concepts

### 3.1 Virtual order tracking

`src/order_tracking.py` is the heaviest module in the repo. `OrderTracker.process_stream` replays a `.dbn.zst` file message by message, maintaining a `Market` of per-instrument `Book` objects. At scheduled points during each trading day it spawns `VirtualOrder` instances — simulated passive limit orders that are never actually sent, but whose queue position is tracked exactly against real order flow.

Each virtual order records:

- Entry context (`entry_time`, `price`, `side`, best bid/ask at entry)
- Execution context (best bid/ask at execution, `fill_price`)
- Post-trade BBO snapshots at **each configured markout window** (1 ms … 60 s)
- A rolling **LOB sequence** and a **toxicity sequence** covering the order's lifetime
- Terminal status: `FILLED`, `CENSORED_TIME`, or `CENSORED_END`

Output is written incrementally to Parquet. Large files are split at *empty-market* boundaries (points where the book is empty, i.e. safe cut points) so that chunks can be processed in parallel by a process pool; the split metadata is cached to JSON and reused by the raw backtest engine.

### 3.2 Feature representations

Two families of per-snapshot features, both defined over the reconstructed book:

**Spatial LOB representations** (`features/representation.py`), on a `(2W+1)` price grid centered at the mid-price:

- `moving_window` — signed volume at each tick offset (asks positive, bids negative)
- `market_depth` — cumulative signed volume outward from the center
- `raw_top5` — raw top-5 bid/ask price-size pairs (20 dims)
- `diff_top5` — first differences of the top-5 representation (20 dims)

The whole sequence is projected onto a **single common price grid** (taken from the most recent snapshot) so the time axis is comparable, with zero left-padding to a fixed lookback length.

**Toxicity features** (`features/compose.py`), 10 microstructure scalars per snapshot plus `time_delta_ms` and `queue_position` (12 dims total):

`spread_relative`, `imbalance_tob`, `depth_imbalance`, `weighted_imbalance`, `total_weighted_volume`, `tob_concentration`, `volume_cv`, `microprice_offset`, `significant_bid_levels`, `significant_ask_levels`.

`ComposeTransforms` chains transforms and concatenates along the feature axis. The default dynamic model input is `raw_top5` (20) + toxicity (12) = **32 features per time step**.

### 3.3 Adaptive toxicity labeling

A fill is "toxic" if the market moves against it afterwards — but *how long* after? Rather than fixing a markout horizon, `MarkoutAnalyzer` computes mean post-trade markouts across all configured windows, and `StabilizationWindowSelector` picks the **earliest horizon whose mean markout reaches a configured fraction (default 90%) of the long-run mean, with matching sign**. The sign check avoids selecting a horizon whose markout happens to have similar magnitude but opposite direction.

`ExecutionCompetingRisksLabeler` then classifies each fill:

- Adverse move is measured from the **execution price** to the post-trade mid at the selected window, in bps, normalized by mid at fill.
- The toxicity threshold is the **dynamic half-spread at fill time** (execution-time BBO preferred; entry-time BBO used only for fills under 100 ms; otherwise the row is censored rather than imputed).
- `adverse_move_bps < threshold_bps` → `FAVORABLE_FILL`, else `TOXIC_FILL`.

Missing execution price, entry BBO, or post-trade BBO all fall back to `CENSORED` with `labeling_valid=False`, on the principle that stale market context teaches the model to confidently predict noise.

### 3.4 Models

All backbones share `BaseDeepHitCompetingModel`, which owns everything except the encoder:

- Input projection + parallel residual projection
- Encoder (`encode()`, subclass-specific)
- Scaled dot-product **attention pooling** over the sequence, plus a residual skip from the latest step
- **Cause-specific heads** producing `(batch, num_events, num_time_bins)` logits
- An **auxiliary head** predicting the next-step feature vector (used by the `L3` term)

| Backbone | Encoder |
|---|---|
| `gru` | Multi-layer unidirectional GRU with pre-norm |
| `gru_transformer` | GRU followed by a pre-norm Transformer encoder with learned positional embeddings |
| `transformer` | Transformer encoder only, learned positional embeddings |
| `mamba` | Stack of pre-norm residual Mamba (SSM) blocks; requires `mamba_ssm` + `causal-conv1d` |

`mamba` degrades gracefully: if the optional CUDA extensions are absent, importing the class succeeds but instantiating it raises a clear `ImportError`.

### 3.5 Loss

`notebook_losses.py` implements the Dynamic-DeepHit objective as `L_total = L1 + alpha * L2 + L3`:

- **L1** — negative log-likelihood over the joint `(event, time-bin)` PMF, handling right-censoring via the survival tail.
- **L2** — cause-specific concordance/ranking loss on the CIFs, with an exponential kernel of width `sigma`; the dynamic variant conditions on the observation index (`update_idx`), the static variant does not.
- **L3** — auxiliary next-step feature-prediction loss from the shared representation, weighted by `beta_l3`.

Every term is averaged **per order first, then across orders** (`_order_average`). This matters because a single order contributes many dynamic samples — plain per-sample averaging would silently overweight long-lived orders.

### 3.6 Backtesting

Two engines share the same `BaseStrategy` contract:

- **`BacktestEngine`** — replays the labeled dataset. Fast, deterministic, ideal for threshold sweeps. Optional *lifecycle-aware* mode re-evaluates a resting order at intermediate snapshots (with a stride and evaluation cap), allowing `HOLD` / `CANCEL` decisions rather than a single entry decision.
- **`RawDatabentoBacktestEngine`** — replays the raw MBO feed, rebuilds the book, rebuilds features live via `LiveFeatureBuilder`, tracks real queue position, and applies a **latency provider** so that a decision taken at time *t* only becomes effective at *t + latency*. Supports sequential chunked replay (GPU-safe) or a process pool (CPU strategies).

**Decision logics** (`strategies/deephit.py`):

- `DeepHitToxicCIFDecisionLogic` — single threshold on the toxic CIF at a chosen horizon index.
- `DeepHitThresholdDecisionLogic` — combines conditional toxic probability among predicted fills (`toxic_cif / fill_cif`) with fill-probability, toxic-CIF, and favorable-CIF gates.

Both map to `SUBMIT`/`SKIP` for new orders and `HOLD`/`CANCEL` for open ones. `DeepHitPredictionCache` memoizes CIF/PMF outputs so threshold grids can be swept without recomputing inference.

**Metric** — `ImplementationShortfallMetric` reports shortfall in bps against a reference mid at the selected markout window, and separates:

- *toxic cost* for orders that were submitted and filled,
- *opportunity cost* for orders that were skipped or cancelled but would have filled,
- *fill-adjusted toxic cost* so strategies with different participation rates remain comparable.

`BacktestReport` aggregates these into mean/median/time-weighted summaries, with bootstrap confidence intervals available in the final-evaluation script.

---

## 4. Pipeline

```
download_data.py
      |  raw .dbn.zst (Databento XNAS.ITCH MBO)
      v
build_dataset.py            per-window virtual order tracking -> raw parquet
      v
merge_datasets.py           combine monthly windows -> raw_dataset_*.parquet
      v
label_dataset.py            markout window selection + competing-risks labels
      v                     -> labeled_dataset_*.parquet
preprocess_dynamic_deephit_dataset.py
      |  day-boundary train/val/test split (70% / 85% row targets)
      |  dynamic-sample manifest, horizon censoring, normalizer, discretizer
      v
notebooks/dynamic_models/standardized_dynamic_deephit.ipynb   (training)
      v                     -> checkpoints/{TICKER}/{model_type}_{TICKER}.pt
deephit_bt_grid_search.py   threshold tuning on a held-out prefix
      v
final_backtest_evaluation.py   final metrics vs always-place baseline
raw_latency_sweep.py           latency sensitivity on the raw engine
sweep_analysis.py              sweep aggregation + bootstrap helpers
```

Splitting is always done **on day boundaries** (`best_day_cut`) so no order straddles a split, and normalization statistics are fitted on the training manifest only.

---

## 5. Setup

### Local / CPU

```bash
pip install -r requirements.txt
```

Create a `.env` at the repo root:

```
DATABENTO_API_KEY=<your_key>
WANDB_ENTITY=<optional>
WANDB_API_KEY=<optional>
```

`scripts/download_data.py` ships with `NO_DOWNLOAD_LOCK = True` as a safety flag — it will load a local file if present and refuse to hit the paid API otherwise. Flip it deliberately.

### GPU

```bash
conda env create -f environment-lob.yml   # creates env "lob"
```

The Mamba backbone additionally needs `causal-conv1d` and `mamba-ssm`, whose CUDA extensions usually have to be **built from source** — the prebuilt wheels and the default system compiler on many machines will not work. In practice this means a modern toolchain (GCC 13 + CUDA 12.4), `CC`/`CXX`/`CUDAHOSTCXX` pointed at it, `TORCH_CUDA_ARCH_LIST` pinned to the target GPU (e.g. `7.0` for V100), and installing with `--no-build-isolation --no-cache-dir --no-binary=:all:`. Keeping `MAX_JOBS=1` avoids exhausting memory on shared nodes. Restart the notebook kernel after installing.

Every other part of the codebase runs without these extensions — see §11.

---

## 6. Running

**Dataset pipeline.** Each stage is a plain script driven by CLI flags and/or environment variables (`SYMBOL`, `START_DATE`, `END_DATE`, `N_WORKERS`, `DATASETS_DIR`):

```bash
python scripts/download_data.py
SYMBOL=AAPL START_DATE=2025-10-01 END_DATE=2025-11-01 python scripts/build_dataset.py
python scripts/merge_datasets.py --ticker AAPL --verbose
python scripts/label_dataset.py --ticker AAPL \
    --start-date 2025-10-01 --end-date 2026-01-01 --datasets-dir <datasets_dir>
python scripts/preprocess_dynamic_deephit_dataset.py --ticker AAPL
```

Build one window per month and merge them, or run a single wide window directly. `build_dataset.py` parallelizes internally across raw-file chunks, so one invocation per window already saturates a multi-core machine.

**Training.** Open `notebooks/dynamic_models/standardized_dynamic_deephit.ipynb`, set `MODEL_NAME` to `gru`, `gru_transformer`, `transformer`, or `mamba`, and run it end to end. Checkpoints are written to `checkpoints/{TICKER}/{model_type}_{TICKER}.pt`. Weights & Biases logging is optional and reads credentials from `.env`.

**Grid searches.** The `_arch_grid_search`, `_loss_grid_search`, and `_lr_grid_search` notebooks are parameterized variants of the training notebook — they read a `(config, ticker, seed)` triple from environment variables and are meant to be executed headlessly with papermill, one process per combination:

```bash
CONFIG_JSONL_PATH=model_configs/deephit_configs_param_200k_500k_v1.jsonl \
papermill notebooks/dynamic_models/standardized_dynamic_deephit_arch_grid_search.ipynb out.ipynb
```

Architecture configs are pre-generated by `notebooks/deephit_architecture_budget_planner.ipynb`, which enumerates candidate architectures and keeps only those whose trainable-parameter count falls inside a target budget (e.g. 200k–500k) — so backbones are compared at matched capacity rather than matched hyperparameters. The loss grid sweeps `alpha` and `beta_l3`; the learning-rate grid fixes architecture and one resolved `(alpha, beta_l3)` pair per backbone. Static-model equivalents live under `notebooks/baseline_models/`.

**Backtests:**

```bash
python scripts/deephit_bt_grid_search.py --ticker AAPL --model-type mamba
python scripts/final_backtest_evaluation.py --ticker AAPL
python scripts/raw_latency_sweep.py --ticker AAPL --model-type mamba --mode sequential
```

**Feature ablation.** `notebooks/dynamic_models/toxicity_feature_ablation.ipynb` runs permutation feature importance over the held-out test set for every trained artifact of a ticker — each feature column is shuffled across samples, inference is re-run, and the degradation in the survival metrics is recorded.

---

## 7. Configuration

`src/config.py` holds every shared constant as a frozen dataclass tree, imported as `CONFIG`. Key values:

| Group | Field | Default | Notes |
|---|---|---|---|
| `data` | `price_unit` | `1e9` | Databento fixed-point price scale |
| | `t_max_s` | `10.0` | Observation horizon |
| | `tox_horizon_s` | `1.0` | Default toxicity horizon |
| `features` | `window` | `20` | Half-width `W` of the price grid |
| | `tick_size` | `0.01 * price_unit` | |
| | `representation` | `market_depth` | Default spatial representation |
| `labeling` | `tox_bps` | `0.2` | Unfavorable-fill indicator |
| | `tox_post_trade_move_windows_ms` | `1 … 60000` | 12 markout horizons |
| | `tox_markout_percentage_threshold` | `0.9` | Stabilization fraction |
| | `binning_strategy` | `log` | Time-bin spacing |
| `time_binning` | `n_bins` | `20` | Discrete-time grid |
| `random_seed` | | `4718` | |

Anything used in more than one place belongs here — the module docstring is explicit that the point is to avoid drift between dataset generation, training, and backtesting.

---

## 8. Tests

```bash
pytest tests/
```

Roughly 100 tests covering order tracking and queue mechanics, LOB representations, labeling, dataset building, dynamic-sample construction, loss correctness (including order-level averaging), preprocessing contracts, and the raw backtest engine. `src/` is importable from tests via the repo-root `sys.path` insertion used consistently across scripts and tests.

---

## 9. Notebooks

**Training & tuning** — `dynamic_models/standardized_dynamic_deephit.ipynb` is the reference end-to-end notebook (load → recode events → build dynamic samples → split → normalize → discretize → train → evaluate), with `_quick` as a fast variant and `_arch/_loss/_lr_grid_search` as parameterized sweep variants. `baseline_models/` holds the static (single-observation) counterparts.

**Analysis** — `deephit_architecture_budget_planner.ipynb` (generate parameter-budgeted config grids), `deephit_architecture_result_analysis.ipynb`, `bt_grid_search_analysis.ipynb` (shortfall vs. participation/fill/cancel trade-off), `dynamic_models/inference_latency_comparison.ipynb` (mean/std/p50/p90 latency and throughput across batch sizes for all four backbones), `dynamic_models/toxicity_feature_ablation.ipynb`.

**EDA & demos** — `dataset_generation_eda.ipynb`, `demo_dataset_generation_one_day.ipynb`, `demo_representations_sampling.ipynb`, `demo_dynamic_dataset_inspection.ipynb`, `post_fill_analysis/window_selection.ipynb`, `stock_selection/stock_selector.ipynb` (liquidity/volatility-based universe selection via `yfinance`).

**Reference implementations** — `notebooks/reference_implementations/` contains unmodified companion notebooks from the deep survival analysis monograph in `references/books/` (exponential model, DeepHit single-event, DeepHit competing risks, Dynamic-DeepHit). They are third-party teaching material kept for comparison and are not part of the pipeline.

---

## 10. References

`references/papers/` collects the literature this work builds on, spanning survival analysis (DeepHit and Dynamic-DeepHit, time-dependent concordance, Brier score and IPCW variants), market microstructure and adverse selection, and LOB representation learning for machine learning models. `references/books/` holds the deep survival analysis monograph; `references/unused/` is material that was reviewed but not used.

---

## 11. Notes and caveats

- **Data is not included.** `data/`, `artifacts/`, `checkpoints/`, `logs/`, `reports/`, `results/`, and `wandb/` are all gitignored. Raw MBO data requires a Databento subscription.
- Several scripts carry **hard-coded cluster paths** (e.g. a shared `datasets` directory) as defaults. They are all overridable by CLI flag or environment variable; check the top of each script before running elsewhere.
- `merge_datasets.py` is configured by editing module-level constants (`TICKER`, `PARQUET_FILES`, `OUTPUT_PATH`) rather than by flags alone.
- Dataset building is memory-hungry; `build_dataset.py` documents swap-file setup for single-machine runs and defaults to a large worker count intended for a cluster node.
- The Mamba backbone is optional everywhere. `src/models/__init__.py` and `src/backtest/__init__.py` both use import guards so the rest of the package works without CUDA extensions.

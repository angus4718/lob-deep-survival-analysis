# Dynamic Deep Survival Analysis for Limit Order Execution Under Adverse Selection

---
## Pre-requisites
Read [An Introduction to Deep Survival Analysis Models for Predicting Time-to-Event Outcomes](references\books\An_Introduction_to_Deep_Survival_Analysis_Models_for_Predicting_Time-to-Event_Outcomes.pdf) by George H. Chen (Associate Professor, Heinz College of Information Systems and Public Policy, Carnegie Mellon University). This book provides a comprehensive introduction to deep survival analysis, covering key concepts such as censoring, loss functions, and model architectures. It serves as a foundational resource for understanding the methodologies we will apply in our project.

Specifically, focus on the following sections:
Section 1, Section 2.1-2.3, Section 2.4 (Example 2.6), Section 2.5 (Harrell's concordance index, Brier Score), Section 3.1-3.4, Section 6. Section 6 is especially important as it covers the DeepHit architecture, which we will implement and adapt for our limit order execution prediction task.

---
## Abstract

This project develops deep competing-risks survival models to jointly predict the time until a limit order is executed and the post-fill price outcome. We combine transformer-based sequence encoders with cause-specific survival heads to distinguish favorable fills, toxic fills (adverse selection), and economically terminal price run-away events, and evaluate models with both time-dependent survival metrics and implementation-shortfall backtests. The objective is to produce risk-aware execution models that improve decision-making and reduce execution costs in high-frequency trading (HFT).

---
## Background
In HFT, algorithmic strategies often involve placing limit orders, which may either be executed or canceled. Understanding the time until execution or cancellation is crucial for optimizing trading strategies.

### Limit Order Book
A limit order is an order to buy or sell a security at a specified price or better. It remains active until it is executed, canceled, or expires. On the other hand, a market order is an order to buy or sell immediately at the best available price.

An limit order book (LOB) is a record of all outstanding limit orders in a market. It shows the quantity of buy and sell orders at various price levels. The order book is dynamic and changes as new orders are placed, executed, or canceled.

### Competing Risks
In survival analysis, competing risks refer to situations where multiple types of events can occur, and the occurrence of one type of event prevents the occurrence of another.
In the context of HFT, there are several competing risks:
1. **Favorable Fill**: The limit order is executed and the price remains stable or moves in a favorable direction. This is the ideal outcome.
2. **Toxic Fill (Adverse Selection)**: The limit order is executed but the price moves in an unfavorable direction. This often occurs when the order is "picked off" by an informed counterparty just before a price crash.
3. **Price Run-away**: The price moves away from the limit order (e.g., price rises while trying to buy) by a threshold $\delta$ before execution occurs. While the order technically still exists in the book, it has economically ``died'' because the probability of execution has collapsed. We treat this as a terminal competing event.

### Right-Censoring
If none of the three events occur within a maximum observation window $T_{max}$ (e.g., 10 seconds), the data point is **Right-Censored**. We know the order ``survived'' at least until $T_{max}$, but we do not know its ultimate fate. Although historical data allows us to track a virtual order indefinitely until execution, we impose an artificial censoring horizon. This design choice is driven by these factors:
1. **Decay of Feature Predictive Power**: The predictive validity of the LOB state at time $t_0$ decays rapidly. An execution occurring at $t_0 + 10\text{s}$ is likely driven by market information arriving at $t_0 + \text{s}$, not the initial state at $t_0$. Labeling such distant events as ``Fills'' would introduce significant noise, forcing the model to learn spurious correlations between current LOB shapes and distant future events.
2. **Economic Realism**: In high-frequency market making, orders are dynamic. If a limit order is not executed within a short horizon, the underlying alpha signal is typically considered stale, and the order would be cancelled or repriced in a production environment. Therefore, treating long-duration orders as ``censored'' (outcome unknown/irrelevant) aligns the model with realistic trading constraints.

---
## Literature Review
### Overview of LOB Fill Prediction
The problem of estimating the probability and timing of limit order execution has evolved from parametric Poisson process models to complex deep learning architectures. Below is a summary of key papers in this area:
1.  **[A Deep Learning Approach to Estimating Fill Probabilities in a Limit Order Book](references\papers\Maglaras2022.pdf)** (Maglaras et al., 2022)
    *   **Contribution:** This study represents a foundational shift from classical econometric models to data-driven deep learning. Maglaras et al. utilize Convolutional Neural Networks (CNNs) to extract features from the LOB shape, treating fill probability as a binary classification problem over fixed time horizons.
    *   **Limitation:** The model utilizes a static binary target (Fill vs. No-Fill), ignoring the continuous temporal nature of execution. It fails to distinguish between *when* an order fills (e.g., 10ms vs. 10s), which is critical for high-frequency strategies.

2.  **[Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers](references\papers\Arroyo2024.pdf)** (Arroyo et al., 2024)
    *   **Contribution:** Arroyo et al. advance the field by framing execution as a **Survival Analysis** problem rather than classification. They introduce a hybrid architecture combining CNNs for spatial feature extraction and Transformers for temporal dependencies. Their work demonstrates that survival loss functions (like Cox Partial Likelihood) outperform standard cross-entropy loss for execution tasks.
    *   **Limitation:** While this paper addresses "Time-to-Fill," it treats all fills as homogeneous events. It does not account for **Adverse Selection**—the risk that a fill is immediately followed by a negative price excursion (toxic flow).

3.  **[Attention-Based Reading, Highlighting, and Forecasting of the Limit Order Book](references\papers\Jung2025.pdf)** (Jung & Lee, 2025)
    *   **Contribution:** This paper focuses on **Explainable AI (XAI)** within market microstructure. Jung & Lee propose a pure Attention mechanism to "highlight" specific orders and price levels in the LOB that drive future price movements. Their work validates the use of Attention weights to interpret black-box financial models.
    *   **Limitation:** The primary focus is on forecasting mid-price direction (alpha) rather than the specific mechanics of limit order execution (implementation). It lacks the specific modeling of queue dynamics required for execution algorithms.

### Our Innovation
While the existing literature has successfully applied Deep Learning to LOB data, a critical gap remains in the **joint modeling of execution timing and execution quality**.

Our project introduces the specific innovations below:

#### 1. Methodological: From Binary Classification to Competing Risks
Most prior works, including Maglaras et al. (2022), treat limit order execution as a binary outcome (Fill vs. No-Fill). This conflates two economically opposing scenarios:
*   **Scenario A:** The trader captures the spread (profit).
*   **Scenario B:** The trader is "picked off" by an informed counterparty just before a price crash (loss).

We reject the homogeneity of "Fills." By adopting a **Deep Competing Risks** framework, we explicitly model the joint distribution of **Time-to-Fill** and **Post-Fill Price Stability**. This allows our model to distinguish between *Liquidity Capture* (Favorable Fill) and *Adverse Selection* (Toxic Fill), effectively transforming the model from a simple execution predictor into a risk-aware execution strategist.

#### 2. Architectural: Transformer-Enhanced Feature Extraction
While Arroyo et al. (2024) utilize CNNs, financial time-series data is characterized by "burstiness" and long-range dependencies that Convolutional networks often miss. We replace standard recurrence with a **Transformer Encoder with Multi-Head Self-Attention**.
*   This architecture allows the model to attend to specific historical microstructure events (e.g., a liquidity vacuum occurring 5 seconds prior) that disproportionately impact current survival probabilities.
*   The attention weights provide a layer of interpretability, allowing us to visualize which market features (e.g., depth imbalance vs. trade flow) drive the model's prediction of toxic flow.

#### 3. Evaluation: Economic Utility over Statistical Concordance
A high Concordance Index (C-index) or low Log-Likelihood does not guarantee profitability. A model might be statistically accurate on calm days but fail catastrophically during volatility. We extend the evaluation framework beyond standard survival metrics to include **Implementation Shortfall (IS)** backtesting. By simulating a strategy that switches between Limit and Market orders based on the model's predicted Cumulative Incidence Function (CIF), we demonstrate the tangible economic value of the model in reducing execution costs.

---
## Dataset
We obtain NASDAQ ITCH data from  **Databento** Market-by-Order (MBO) schemas. The fields utilized are listed below:

| Field | Type | Description |
| :--- | :--- | :--- |
| `ts_recv` | `uint64_t` | The capture-server-received timestamp (nanoseconds since UNIX epoch). |
| `ts_event` | `uint64_t` | The matching-engine-received timestamp (nanoseconds since UNIX epoch). |
| `rtype` | `uint8_t` | A sentinel value indicating the record type. Always `160` in the MBO schema. |
| `publisher_id` | `uint16_t` | The publisher ID assigned by Databento, denoting the dataset and venue. |
| `instrument_id` | `uint32_t` | The numeric instrument ID. |
| `action` | `char` | The event action (<u>**A**</u>dd, <u>**C**</u>ancel, <u>**M**</u>odify, clea<u>**R**</u> book, <u>**T**</u>rade, <u>**F**</u>ill, or <u>**N**</u>one). |
| `side` | `char` | The side that initiates the event (<u>**A**</u>sk, <u>**B**</u>id, or <u>**N**</u>one). |
| `price` | `int64_t` | The order price where every 1 unit corresponds to 1e-9. |
| `size` | `uint32_t` | The order quantity. |
| `channel_id` | `uint8_t` | The channel ID assigned by Databento as an incrementing integer starting at zero. |
| `order_id` | `uint64_t` | The order ID assigned by the venue. |
| `flags` | `uint8_t` | A bit field indicating event end, message characteristics, and data quality. |
| `ts_in_delta` | `int32_t` | The matching-engine-sending timestamp expressed as nanoseconds before `ts_recv`. |
| `sequence` | `uint32_t` | The message sequence number assigned at the venue. |

For a complete reference and detailed definitions, please consult the [Databento MBO Documentation](https://databento.com/docs/schemas-and-data-formats/mbo).

---
## Feature Engineering
To extract meaningful signals regarding execution probability and adverse selection risk, we transform the raw LOB updates into a set of stationary, informative features.

We categorize our feature set into three distinct levels: **Static State**, **Dynamic Flow**, and **Execution Context**.

### 1. Level-I: Basic LOB State (Static)
These features capture the shape of the book at a specific snapshot $t$. We utilize the top 10 levels of the LOB.

*   **Mid-Price ($P_{mid}$):** The reference price, defined as $\frac{P_{ask}^{(1)} + P_{bid}^{(1)}}{2}$.
*   **Bid-Ask Spread ($S_t$):** The cost of immediate liquidity, $P_{ask}^{(1)} - P_{bid}^{(1)}$. A widening spread often indicates uncertainty or informed trading.
*   **Price Differences ($\Delta P$):** The relative distance between price levels. Since absolute prices are non-stationary, we use differences: $P_{ask}^{(i)} - P_{ask}^{(i-1)}$.
*   **Depth Imbalance ($\rho_t$):** A strong predictor of short-term price direction. It measures the pressure on the bid vs. ask side:
    $\rho_t = \frac{\sum_{i=1}^{k} V_{bid}^{(i)} - \sum_{i=1}^{k} V_{ask}^{(i)}}{\sum_{i=1}^{k} V_{bid}^{(i)} + \sum_{i=1}^{k} V_{ask}^{(i)}}$
    *   *Intuition:* If $\rho_t > 0$, buy pressure is high. For a buy limit order, this increases the risk of **Price Run-away** (price moving up away from the order).

### 2. Level-II: Dynamic Flow (Kinematic)
Static features ignore the *rate of change*. We incorporate flow features to capture the "velocity" of the market, which is critical for predicting **Toxic Fills**.

*   **Order Flow Imbalance (OFI):** Based on the work of Cont et al., OFI measures the net flow of limit orders at the best touch. It captures the intent of market participants (e.g., cancellations on the bid side vs. new additions on the ask side).
*   **Trade Flow Imbalance (TFI):** The net volume of market orders hitting the bid vs. lifting the ask over a rolling window $\Delta t$.
    *   *Intuition:* A surge in aggressive sell orders (negative TFI) while our buy limit order is active significantly increases the probability of **Adverse Selection**.
*   **Realized Volatility ($\sigma_t$):** Calculated as the standard deviation of mid-price returns over the past $n$ ticks. High volatility increases the probability of execution (both favorable and toxic) but reduces the probability of the order resting peacefully.

### 3. Level-III: Execution Context (Conditional)
Unlike standard forecasting models, our Survival framework requires features specific to the *agent's* order, not just the market.

*   **Simulated Queue Position ($Q_{pos}$):**
    Because exchanges operate on a Price-Time Priority (FIFO) basis, the probability of a fill depends heavily on where the order sits in the queue.
    *   We simulate $Q_{pos}$ as a normalized value $[0, 1]$.
    *   $Q_{pos} = 0$: Front of the queue (next to be filled).
    *   $Q_{pos} = 1$: Back of the queue (last to be filled).
    *   *Significance:* An order at the back of the queue is protected from "noise" fills but highly exposed to **Adverse Selection** (only getting filled when the entire level is wiped out by a large informed trade).

### 4. Data Preprocessing & Normalization
*   **Z-Score Normalization:** Applied to Volume, Spread, and Volatility features to center the mean at 0 and variance at 1.
*   **Log-Transformation:** Applied to raw Volume levels before normalization, as volume data typically follows a power-law distribution.
*   **Stationarity Checks:** All price inputs are converted to returns or relative differences to ensure stationarity, preventing the model from learning specific price levels (e.g., "Buy at $150") rather than market dynamics.

---
## Model Architecture: The Dynamic-DeepHit Framework
This section is adapted from Section 6 of Prof. George H. Chen's book, with modifications specific to the Limit Order Book execution context.

To address the stochastic nature of LOB dynamics and the existence of multiple competing outcomes (Favorable Fill, Toxic Fill, Run-away), we employ the **Dynamic-DeepHit** architecture. This framework extends standard survival analysis by handling **longitudinal data** (time-series) and **competing risks** simultaneously.

Unlike static models that view a trade only at its initiation ($t=0$), this architecture updates its survival predictions dynamically as new market data arrives, utilizing a history of variable length $m$.

The architecture is composed of four functional blocks: the **Longitudinal Encoder (RNN)**, the **Temporal Attention Module**, the **Cause-Specific Sub-networks**, and a **Multi-Objective Loss Function**.

### 1. Input Representation
Let $X_i$ represent the time-series of LOB states for a specific order $i$. The input is defined as a sequence of observations and timestamps:
$X_i = \left( (u^{(1)}, v^{(1)}), (u^{(2)}, v^{(2)}), \dots, (u^{(m)}, v^{(m)}) \right)$
*   **$u^{(p)}$ (Features):** The raw LOB features (Price levels, Volumes, Order Flow) at step $p$.
*   **$v^{(p)}$ (Timestamps):** The elapsed time since the initial observation.
*   **$m$:** The number of time steps observed so far (variable length).

### 2. Longitudinal Encoder (RNN)
To capture the temporal evolution of the market, we utilize a Recurrent Neural Network (RNN) or LSTM. This network processes the history of the LOB **excluding the current step** $m$.

For each step $p \in \{1, \dots, m-1\}$, the RNN receives the feature vector $u^{(p)}$ and the time interval to the next step $\Delta v = v^{(p+1)} - v^{(p)}$.
$\tilde{u}^{(p)} = \text{RNN}\left( u^{(p)}, \Delta v \right)$
Here, $\tilde{u}^{(p)}$ represents the hidden state vector encoding the market dynamics at historical step $p$.

### 3. Temporal Attention Mechanism
A key innovation of Dynamic-DeepHit is the attention mechanism (Figure 7 in the reference text). This module determines which historical market states are most relevant to the **current** market state $u^{(m)}$.

We compute a context vector $\bar{s}$ by taking a weighted sum of historical hidden states:
$\bar{s} = \sum_{p=1}^{m-1} \alpha_p \tilde{u}^{(p)}$

The attention weights $\alpha_p$ are derived via a softmax function applied to a feed-forward alignment network $f_{att}$:
$\alpha_p = \text{Softmax}\left( f_{att}(\tilde{u}^{(p)}, u^{(m)}; \theta_{att}) \right)$

This allows the model to "attend" to specific historical regimes (e.g., a liquidity vacuum 10 steps ago) that are structurally similar or predictive of the current LOB state $u^{(m)}$.

### 4. Cause-Specific Estimation Heads
The final prediction is generated by combining the current market state with the historical context. The vector $(u^{(m)}, \bar{s})$ is concatenated and fed into $K$ separate Multi-Layer Perceptrons (MLPs), where $K$ is the number of competing risks (e.g., $K=3$ for Favorable, Toxic, Run-away).

Each MLP outputs a vector of size $L$ (discretized time horizons). The final layer is a joint Softmax applied across all $K$ events and $L$ time intervals to ensure the probability mass function (PMF) sums to 1 (including the probability of survival beyond the maximum horizon).

The output is the conditional probability of event $k$ occurring at time interval $\tau$, given the history up to step $m$:
$P(T = \tau, E = k \mid X^{(\le m)})$

### 5. Multi-Objective Loss Function
Training is performed using a specialized loss function $L_{Total}$ composed of three terms, as defined in Equation (83) of the reference text:

$L_{Total} = L_{NLL} + \eta L_{Rank} + \gamma L_{Forecasting}$

1.  **Negative Log-Likelihood ($L_{NLL}$):**
    Maximizes the likelihood of the true event occurring at the true time. For censored data (unfilled orders), it maximizes the probability of surviving past the censoring time.

2.  **Ranking Loss ($L_{Rank}$):**
    A differentiable approximation of the Concordance Index. It penalizes the model if it assigns a lower risk score to an order that filled quickly compared to an order that filled slowly (or not at all). This ensures the model correctly ranks the relative urgency of different market states.

3.  **Auxiliary Forecasting Loss ($L_{Forecasting}$):**
    *Specific to Dynamic-DeepHit.* This term forces the RNN to learn meaningful temporal dynamics by requiring it to predict the **next step's** raw input $u^{(p+1)}$ given the current state $\tilde{u}^{(p)}$.
    $L_{Forecasting} = \sum_{p=1}^{m-1} \Vert \hat{u}^{(p+1)} - u^{(p+1)} \Vert ^2$
    This regularization prevents overfitting and ensures the RNN captures the underlying generative process of the Limit Order Book.

---
## Implementation Details and Training Strategy
This section is adapted from Section 6 of Prof. George H. Chen's book, with modifications specific to the Limit Order Book execution context.

To effectively deploy the Dynamic-DeepHit architecture for LOB execution, several implementation-specific preprocessing steps and training protocols are required. These ensure the model does not overfit to the length of the training sequences and correctly handles the heavy-tailed nature of market event times.

### 1. Time Horizon Discretization
The DeepHit framework requires the continuous time-to-event scale to be discretized into $L$ intervals. Given that LOB reaction times often follow a power-law distribution (many fast fills, few very long waits), a linear time grid is suboptimal.

Instead, we employ **Quantile-Based Discretization**. We select $L$ cut-off points $\tau_1, \tau_2, \dots, \tau_L$ such that the training data is distributed approximately evenly across these bins. The $l$-th output of the network corresponds to the probability that the event occurs in the interval $(\tau_{l-1}, \tau_l]$.

### 2. Data Augmentation and Sub-Sequence Sampling
A critical challenge in dynamic survival analysis is **sampling bias**, as highlighted in the Dynamic-DeepHit methodology. If the model is trained only on completed trade sequences (where the final step $M$ is the moment immediately preceding the fill), it may learn to rely on "terminal" features (e.g., the price crossing the spread) rather than learning the temporal evolution leading up to that event.

To mitigate this, we utilize the **Augmented Training Points** strategy:
For every training sequence $X_i$ of length $M_i$, we generate $M_i$ separate training instances:
1.  **Sub-sequence 1:** Input $x^{(\le 1)}$, Target $Y_i$ (adjusted for elapsed time).
2.  **Sub-sequence 2:** Input $x^{(\le 2)}$, Target $Y_i$ (adjusted for elapsed time).
3.  ...
4.  **Sub-sequence $M_i$:** Input $x^{(\le M_i)}$, Target $Y_i$.

During each training epoch, we randomly sample one sub-sequence length per order. This forces the RNN to learn robust representations at *every* stage of the order's lifespan, not just at the moment of execution.

### 3. Hyperparameter Configuration
The architecture is instantiated with the following hyperparameters, tuned via validation performance on the C-index:

*   **Longitudinal Encoder:** A Long Short-Term Memory (LSTM) network with a hidden state dimension of $d_{hidden}$.
*   **Attention Network:** A single-layer feed-forward network mapping the concatenation of history and current state to a scalar attention weight.
*   **Cause-Specific MLPs:** For each of the $K=3$ competing risks (Favorable, Toxic, Run-away), we use an MLP with hyperparameters to be tuned.
*   **Loss Weights:**
    *   $\eta$ (Ranking Loss weight)
    *   $\gamma$ (Forecasting Loss weight)
    *   $\sigma$ (Ranking Softness)

### 4. Optimization
To be decided.

---
## Baseline Models
To benchmark the performance of our Dynamic-DeepHit model, we implement the following baseline models:
1.  **Static-DeepHit:** A simplified version of DeepHit that only considers the initial LOB state at order placement ($t=0$) without any temporal dynamics. This model serves to isolate the value added by the longitudinal encoder and attention mechanism.
2. **Dynamic-DeepSurv:** An adaptation of the DeepSurv model for dynamic data, which uses an RNN to encode the time-series but predicts a single hazard function without competing risks. This allows us to assess the importance of modeling multiple event types.

---
## Evaluation Metrics
We evaluate model performance using both statistical survival metrics and economic utility measures.
### 1. Statistical Metrics
*   **Concordance Index (C-index):** Measures the model's ability to correctly rank order survival times.
*   **Brier Score:** Assesses the accuracy of predicted survival probabilities over time.
*   **Cause-Specific Cumulative Incidence Functions (CIFs):** Evaluates the accuracy of predicted probabilities for each competing risk over time.
### 2. Economic Utility Metrics
*   **Implementation Shortfall (IS):** We simulate a trading strategy that uses the model's CIF predictions to decide between placing a limit order or switching to a market order. The IS is calculated as the difference between the execution price and the benchmark price, averaged over all trades.
*   **Cost Reduction Percentage:** The percentage reduction in execution costs compared to a naive strategy (e.g., always using market orders).

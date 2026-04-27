module purge
module load gcc/13.3.1-p20240614
module load anaconda3
module load cuda/12.6.1
conda deactivate
conda activate /ocean/projects/cis260122p/hwang71/envs/backtest_mamba
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CC=$(which gcc)
export CXX=$(which g++)
cd /ocean/projects/cis260122p/hwang71
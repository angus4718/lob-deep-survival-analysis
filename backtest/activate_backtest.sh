module purge
module load anaconda3
conda deactivate
conda activate /ocean/projects/cis260122p/hwang71/envs/backtest310
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
cd /ocean/projects/cis260122p/hwang71

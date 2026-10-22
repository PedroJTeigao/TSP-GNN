# Ativar com: source env.sh   (só afeta este terminal)
source /scratch/pcanaste/projeto/gnn-tsp/.venv/bin/activate
export PATH=/scratch/pcanaste/node/bin:/scratch/pcanaste/npm-global/bin:$PATH
export TMPDIR=/scratch/pcanaste/projeto/gnn-tsp/tmp   # o /tmp do sistema é pequeno
export OMP_NUM_THREADS=8                               # não ocupar os 24 CPUs
export PIP_CACHE_DIR=/scratch/pcanaste/projeto/gnn-tsp/tmp/pip-cache
export PYTHONPATH=/scratch/pcanaste/projeto/gnn-tsp/tsp/src:/scratch/pcanaste/projeto/gnn-tsp/milp/src${PYTHONPATH:+:$PYTHONPATH}   # para "import tsp_gnn" e "import fcnf_gnn"

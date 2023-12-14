#!/bin/bash

#SBATCH -A BCS20003
#SBATCH -J meshnet_origin         # Job name
#SBATCH -o meshnet_origin.o%j     # Name of stdout output file
#SBATCH -e meshnet_origin.e%j     # Name of stderr error file
#SBATCH -p gpu-a100-small              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu

# Start at frontera
source venvs/start_venv_ls6_torch20.sh

cd meshnet
export CASE="pipe-h5"
python3 train.py \
--data_path="/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/${CASE}/" \
--model_path="/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/models/${CASE}/" \
--ntraining_steps=5000000 \
--nsave_steps=5000

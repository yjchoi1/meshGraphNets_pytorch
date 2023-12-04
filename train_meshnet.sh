#!/bin/bash

#SBATCH -A BCS20003
#SBATCH -J meshnet_origin         # Job name
#SBATCH -o meshnet_origin.o%j     # Name of stdout output file
#SBATCH -e meshnet_origin.e%j     # Name of stderr error file
#SBATCH -p rtx              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=yj.choi@utexas.edu


source start_venv_frontera.sh
python3 train.py

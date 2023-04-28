#!/bin/bash
#
#SBATCH --job-name=pf3000 # Job name for allocation
#SBATCH --output=slurm/%j.log # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition=ampere # Partition/Queue to submit job
#SBATCH --gres=gpu:a100:3 # Number of GPUs
#SBATCH --mem=80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12 # Number of tasks per node
#SBATCH --mail-user=cwang31@slac.stanford.edu # Receive e-mail from slurm
#SBATCH --mail-type=ALL # Type of e-mail from slurm; other options are: Error, Info.
#SBATCH --time=2-00:00:00
#
 
# "-u" flushes print statements which can otherwise be hidden if mpi hangs
# "-m mpi4py.run" allows mpi to exit if one rank has an exception
## mpirun python -u -m mpi4py.run /reg/data/ana03/scratch/cwang31/pf/mpi.train.fast.Rayonix.cat.att.py
python train.fast.cat.att.py

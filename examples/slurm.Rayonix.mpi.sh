#!/bin/bash
#
#SBATCH --job-name=pf.Raynoix # Job name for allocation
#SBATCH --output=slurm/%j.log # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition=psanagpuq # Partition/Queue to submit job
#SBATCH --gres=gpu:1080ti:1 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # Number of tasks per node
#SBATCH --mail-user=cwang31@slac.stanford.edu # Receive e-mail from slurm
#SBATCH --mail-type=ALL # Type of e-mail from slurm; other options are: Error, Info.
#
 
# "-u" flushes print statements which can otherwise be hidden if mpi hangs
# "-m mpi4py.run" allows mpi to exit if one rank has an exception
mpirun python -u -m mpi4py.run /reg/data/ana03/scratch/cwang31/pf/mpi.train.fast.Rayonix.multiclass.py

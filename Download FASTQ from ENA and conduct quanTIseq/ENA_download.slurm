#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Mon Feb 08 2021 21:27:01 GMT+1100 (AEDT)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="ENA_download_test"

# The project ID which this job should run under:
#SBATCH --account="punim1382"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1



# Use this email address:
#SBATCH --mail-user=yizhem@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=10-1:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Load required modules
module load gcc/8.3.0
module load singularity/3.6.3
module load python/3.8.2
module load web_proxy

# Run the job from the directory where it was launched (default)
python ENAdownload_paths.py
sh Download.sh

ENA_download.slurm (END)


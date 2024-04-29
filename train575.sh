#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=16:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=instruction    # gpu node(s)
#SBATCH --account=s2024.com_s.672.1
#SBATCH --job-name="train575"
#SBATCH --mail-user=gamerhritik@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
module purge
cd /home/hritikz/projects/575/Data/CMGAN2/

source bin/activate

pwd
python --version
which python
which pip
cd src/
python evaluation.py --test_dir /work/classtmp/hritikz/VCTK-3/test/ > test_output.txt

#test a.py
#python a.py > b.txt

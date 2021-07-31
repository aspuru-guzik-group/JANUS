#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128000M               # memory (per node)
#SBATCH --time=23:59:59
#SBATCH --job-name GA


module --force purge
module load nixpkgs/16.09
module load gcc/7.3.0
module load rdkit/2019.03.4

module load scipy-stack/2019b

source ~/ENV_2/bin/activate

python3 GA.py
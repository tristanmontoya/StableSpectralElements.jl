#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name dgmulti_p9_upwind

module load NiaEnv/2019b julia/1.7

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=4

scheme=DGMulti
form=SkewSymmetricMapping

cd /home/z/zingg/tmontoya/scratch/CLOUD.jl/drivers

julia --project=.. --threads 8 advection_2d.jl -b 0.0025 -m 0.2 -p 9 -r 9 -l 1.0 -M 2 -g 4 -s $scheme -f $form -i CarpenterKennedy2N54 -n 50
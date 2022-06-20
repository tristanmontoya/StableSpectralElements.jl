#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name collapsedsem_hrefine

module load NiaEnv/2019b julia/1.7

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/home/z/zingg/tmontoya/scratch/CLOUD.jl/.mplconfig/

scheme=CollapsedModal
form=SplitConservationForm

cd /home/z/zingg/tmontoya/scratch/CLOUD.jl/drivers

julia --project=.. --threads 4 advection_2d.jl -C 0.0005 -m 0.2 -p 4 -r 4 -l 0.0 -M 2 -g 6 -s $scheme -f $form -i DP8 -n 50 &
julia --project=.. --threads 4 advection_2d.jl -C 0.0005 -m 0.2 -p 4 -r 4 -l 1.0 -M 2 -g 6 -s $scheme -f $form -i DP8 -n 50 &
julia --project=.. --threads 16 advection_2d.jl -C 0.0005 -m 0.2 -p 9 -r 9 -l 0.0 -M 2 -g 6 -s $scheme -f $form -i DP8 -n 50 &
julia --project=.. --threads 16 advection_2d.jl -C 0.0005 -m 0.2 -p 9 -r 9 -l 1.0 -M 2 -g 6 -s $scheme -f $form -i DP8 -n 50
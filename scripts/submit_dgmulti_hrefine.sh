#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name dgmulti_hrefine_p2_central

module load NiaEnv/2019b julia/1.7

source ~/.virtualenvs/tristan/bin/activate

p=2
scheme=DGMulti
form=WeakConservationForm


cd /home/z/zingg/tmontoya/scratch/CLOUD.jl/drivers

julia --project=.. --threads 16 advection_2d.jl -m 0.2 -p $p -r $p -l 0.0 -M 2 -g 5 -s $scheme -f $form > out.txt

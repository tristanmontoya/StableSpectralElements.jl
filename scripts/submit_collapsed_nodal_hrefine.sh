#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name collapsed_nodal_hrefine_p10

module load NiaEnv/2019b julia/1.7

source ~/.virtualenvs/tristan/bin/activate

p=10
scheme=CollapsedSEM
form=SplitConservationForm


cd /home/z/zingg/tmontoya/scratch/CLOUD.jl/drivers

julia --project=.. --threads 16 advection_2d.jl -p $p -r $p -M 2 -g 4 -s $scheme -f $form > out.txt


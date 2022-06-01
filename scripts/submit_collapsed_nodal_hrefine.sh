#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --job-name collapsed_nodal_hrefine

module load NiaEnv/2019b julia/1.7 python/3.8

source ~/.virtualenvs/tristan/bin/activate

p=2
scheme=CollapsedSEM
form=SplitConservationForm


cd /home/z/zingg/tmontoya/scratch/CLOUD.jl/drivers


for M in 2 4 8 16
do
	julia --project=.. advection_2d.jl -p $p -M $M -s $scheme -f $form > out.txt &
done


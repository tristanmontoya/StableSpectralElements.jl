#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=6:00:00
#SBATCH --job-name scaling_test

module load NiaEnv/2019b 
cd /scratch/z/zingg/tmontoya/ScalingTests/scripts
export OPENBLAS_NUM_THREADS=1

for nthreads in 1 2 4 5 8 10 20 40 80
do 
    julia --project=.. --threads $nthreads --check-bounds=no -e "using StableSpectralElements; scaling_test_euler_2d(4,16,"./results/p4M16/", Threaded())" > screen.txt
done
#!/bin/zsh

julia --project=.. --threads 1 linear_advection.jl --path ../results/refinement_test_2d/ -b 0.25 -m 0.2 -p 4 -r 4 -l 1.0 -M 2 -g 6 -e Quad -s NodalTensor -f SkewSymmetricMapping -i CarpenterKennedy2N54 -n 50
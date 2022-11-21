#!/usr/bin/env bash

julia --project=.. --threads 1 linear_advection_refinement.jl --path ../results/refinement_test_3d/ -b 0.25 -m 0.1 -p 4 -r 4 -l 1.0 -M 2 -g 6 -e Tet -s ModalMulti -f SkewSymmetricMapping -i CarpenterKennedy2N54 -n 50 
#!/usr/bin/env bash

refinement_path="../results/refinement_test/"
scheme="NodalTensor"
elem_type="Quad"
mapping_form="SkewSymmetricMapping"

polynomial_degrees=(4 9)
upwind_parameters=(0.0 1.0)

for p in ${polynomial_degrees[@]}; do
    for lambda in ${upwind_parameters[@]}; do
        julia --project=.. --threads 1 linear_advection_refinement.jl --path $refinement_path -b 0.1 -m 0.2 -p $p -r $p -l $lambda -M 2 -g 4 -e $elem_type -s $scheme -f $mapping_form -i CarpenterKennedy2N54 -n 100 &
    done
done
wait

julia --project=.. linear_advection_refinement_analysis.jl ${polynomial_degrees[@]} -e $elem_type -s $scheme -f $mapping_form --path $refinement_path
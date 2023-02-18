#!/usr/bin/env bash

refinement_path="../results/advection_hrefine_feb17_tri_nodalmulti/"
scheme="NodalMulti"
elem_type="Tri"
mapping_form="SkewSymmetricMapping"

polynomial_degrees=(4)
mapping_degrees=(4)
upwind_parameters=(0.0 1.0)

for i in ${!polynomial_degrees[@]}; do
    for lambda in ${upwind_parameters[@]}; do
        julia --project=.. --threads 1 advection_refinement.jl --path $refinement_path --CFL 0.01 -m 0.05 -p ${polynomial_degrees[i]} -l ${mapping_degrees[i]} --lambda $lambda -M 2 -g 3 -e $elem_type -s $scheme -f $mapping_form -i CarpenterKennedy2N54 --mass_solver WeightAdjustedSolver -n 100 &
    done
done
wait
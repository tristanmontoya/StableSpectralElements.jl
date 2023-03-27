#!/usr/bin/env bash

refinement_path="../results/advection_tetp4_32_new/"
scheme="NodalTensor"
elem_type="Tet"
mapping_form="SkewSymmetricMapping"

polynomial_degrees=(4)
mapping_degrees=(3)
upwind_parameters=(0.0 1.0)

for i in ${!polynomial_degrees[@]}; do
    for lambda in ${upwind_parameters[@]}; do
        julia --project=.. advection_refinement.jl --path $refinement_path --CFL 0.01 -m 0.05 -p ${polynomial_degrees[i]} -l ${mapping_degrees[i]} --lambda $lambda -M 32 -g 1 -e $elem_type -s $scheme -f $mapping_form -i CarpenterKennedy2N54 --mass_solver CholeskySolver -n 100 --load_from_file &
    done
done
wait

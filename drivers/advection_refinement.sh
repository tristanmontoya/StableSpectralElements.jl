#!/usr/bin/env bash

refinement_path="../results/advection_hrefine/"
scheme="ModalTensor"
elem_type="Tri"
mapping_form="SkewSymmetricMapping"

polynomial_degrees=(4 9)
upwind_parameters=(0.0 1.0)

for p in ${polynomial_degrees[@]}; do
    for lambda in ${upwind_parameters[@]}; do
        julia --project=.. --threads 1 advection_refinement.jl --path $refinement_path -b 0.005 -m 0.2 -p $p -l $p --lambda $lambda -M 2 -g 3 -e $elem_type -s $scheme -f $mapping_form -i CarpenterKennedy2N54 -n 100 --overwrite &
    done
done
wait
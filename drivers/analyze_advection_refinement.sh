#!/usr/bin/env bash

refinement_path="../results/advection_hrefine_feb16_tri2/"
scheme=("NodalTensor" "NodalMulti")
elem_type=("Tri" "Tri")
mapping_form=("SkewSymmetricMapping" "SkewSymmetricMapping")
name=("Tensor-product" "Multidimensional")

polynomial_degrees=(4)

julia --project=.. analyze_advection_refinement.jl -p ${polynomial_degrees[@]} -e ${elem_type[@]} -s ${scheme[@]} -f ${mapping_form[@]} -n ${name[@]} --path $refinement_path --weight_adjusted --no_p
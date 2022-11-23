#!/usr/bin/env bash

refinement_path="../results/advection_hrefine/"
scheme=("NodalTensor" "ModalTensor")
elem_type=("Tri" "Tri")
mapping_form=("SkewSymmetricMapping" "SkewSymmetricMapping")
name=("Nodal" "Modal")

polynomial_degrees=(4 9)

julia --project=.. analyze_advection_refinement.jl -p ${polynomial_degrees[@]} -e ${elem_type[@]} -s ${scheme[@]} -f ${mapping_form[@]} -n ${name[@]} --path $refinement_path
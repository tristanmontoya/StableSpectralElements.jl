#!/usr/bin/env bash
analysis_path="../results/spectral_radius_feb_19/"
scheme="ModalTensor"
elem_type="Tri"
mapping_form="SkewSymmetricMapping"
nev=1

julia --project=.. --threads 1 advection_analysis.jl --path $analysis_path -m 0.05 -p {3..16} -l {3..16} -M 2 -e $elem_type -s $scheme -f $mapping_form -n $nev

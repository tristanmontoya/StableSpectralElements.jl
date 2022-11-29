#!/usr/bin/env bash
analysis_path="../results/advection_analysis_221128/"
scheme="ModalTensor"
elem_type="Tri"
mapping_form="SkewSymmetricMapping"
nev=1

julia --project=.. --threads 1 advection_analysis.jl --path $analysis_path -m 0.2 -p {3..13} -l {3..13} -M 4 -e $elem_type -s $scheme -f $mapping_form -n $nev
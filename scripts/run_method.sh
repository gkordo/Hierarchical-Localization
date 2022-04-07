#!/bin/bash

OUTPUTS=$1
METHOD=$2
IM_SIZE=$3
REFINEMENT=$4

if [[ $METHOD == "netvlad"  ||  $METHOD == "dns"  ||  $METHOD == "geoloc" ]]; then
    echo "Selected method: $2"
else
    echo "Error: method does not exists. Please, select one of the available methods."
    echo "Available methods: netvlad, dns, geoloc"
    exit
fi

if [[ $REFINEMENT == "none" ]]; then
    python -m hloc.pipelines.Aachen.pipeline --outputs $OUTPUTS --retrieval $METHOD --im_size $IM_SIZE
elif [[ $REFINEMENT == "W" ]]; then
    python -m hloc.pipelines.Aachen.pipeline --outputs $OUTPUTS --retrieval $METHOD --im_size $IM_SIZE --whitening
elif [[ $REFINEMENT == "MS" ]]; then
    python -m hloc.pipelines.Aachen.pipeline --outputs $OUTPUTS --retrieval $METHOD --im_size $IM_SIZE --multiscale '[2**(1/2), 1, 1/2**(1/2)]'
elif [[ $REFINEMENT == "QE" ]]; then
    python -m hloc.pipelines.Aachen.pipeline --outputs $OUTPUTS --retrieval $METHOD --im_size $IM_SIZE --query_expansion 5
elif [[ $REFINEMENT == "TD" ]]; then
    python -m hloc.pipelines.Aachen.pipeline --outputs $OUTPUTS --retrieval $METHOD --im_size $IM_SIZE --use_todaygan
else
    echo "Error: refinement does not exists. Please, select one of the available refinements."
    echo "Available refinements: none, W, MS, QE, TD"
    exit
fi
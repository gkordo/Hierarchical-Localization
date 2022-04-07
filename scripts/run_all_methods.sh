#!/bin/bash

OUTPUTS=$1

bash scripts/run_method_all_sizes.sh $OUTPUTS dns
bash scripts/run_method_all_refinements.sh $OUTPUTS dns 450

bash scripts/run_method_all_sizes.sh $OUTPUTS geoloc
bash scripts/run_method_all_refinements.sh $OUTPUTS geoloc 380

bash scripts/run_method.sh $OUTPUTS $METHOD 1024 none
bash scripts/run_method_all_refinements.sh $OUTPUTS netvlad 1024

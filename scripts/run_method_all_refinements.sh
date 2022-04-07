#!/bin/bash

OUTPUTS=$1
METHOD=$2
IM_SIZE=$3

bash scripts/run_method.sh $OUTPUTS $METHOD $IM_SIZE W
bash scripts/run_method.sh $OUTPUTS $METHOD $IM_SIZE MS
bash scripts/run_method.sh $OUTPUTS $METHOD $IM_SIZE QE
bash scripts/run_method.sh $OUTPUTS $METHOD $IM_SIZE TD

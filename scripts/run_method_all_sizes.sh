#!/bin/bash

OUTPUTS=$1
METHOD=$2

bash scripts/run_method.sh $OUTPUTS $METHOD 224 none
bash scripts/run_method.sh $OUTPUTS $METHOD 300 none
bash scripts/run_method.sh $OUTPUTS $METHOD 380 none
bash scripts/run_method.sh $OUTPUTS $METHOD 450 none
bash scripts/run_method.sh $OUTPUTS $METHOD 600 none

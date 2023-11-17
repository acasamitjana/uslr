#!/usr/bin/env bash

if  [ $# -eq 0 ]; then
    echo ""
    echo "********************************"
    echo "Running all subjects in BIDS_DIR"
    echo "********************************"
    echo ""
    python ../pipeline/preprocess/synthseg.py
    python ../pipeline/preprocess/bias_field.py
else
        echo ""
    echo "*****************************"
    echo "Running subject(s) " $@
    echo "*****************************"
    echo ""
    python ../pipeline/preprocess/synthseg.py --subjects $@
    python ../pipeline/preprocess/bias_field.py --subjects $@
fi







#!/usr/bin/env bash

if  [ $# -eq 0 ]; then
    echo ""
    echo "********************************"
    echo "Running all subjects in BIDS_DIR"
    echo "********************************"
    echo ""
#    python ../pipeline/preprocess/synthseg.py
    python ../pipeline/preprocess/bias_field.py

    python ../pipeline/uslr_linear/initialize_graph.py
    python ../pipeline/uslr_linear/compute_graph.py
    python ../pipeline/uslr_linear/compute_template.py

    python ../pipeline/uslr_nonlinear/initialize_graph.py
    python ../pipeline/uslr_nonlinear/compute_graph.py --num_cores 2
    python ../pipeline/uslr_nonlinear/compute_template.py
    python ../pipeline/uslr_nonlinear/register_template.py

    python ../pipeline/postprocess/par_trans_tp.py
    python ../pipeline/postprocess/long_segmentation.py


else
    echo ""
    echo "*****************************"
    echo "Running subject(s) " $@
    echo "*****************************"
    echo ""
    python ../pipeline/preprocess/synthseg.py --subjects $@
    python ../pipeline/preprocess/bias_field.py --subjects $@

    python ../pipeline/uslr_linear/initialize_graph.py --subjects $@
    python ../pipeline/uslr_linear/compute_graph.py --num_cores 2 --subjects $@
    python ../pipeline/uslr_linear/compute_template.py --subjects $@

    python ../pipeline/uslr_nonlinear/initialize_graph.py --subjects $@
    python ../pipeline/uslr_nonlinear/compute_graph.py --num_cores 2 --subjects $@
    python ../pipeline/uslr_nonlinear/compute_template.py --subjects $@
fi







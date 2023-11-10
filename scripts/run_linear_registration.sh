#!/usr/bin/env bash

if  [ $# -eq 0 ]; then
    echo "Running all subjects in IMAGES_DIR"
    python ../pipeline/uslr_linear/initialize_graph.py
    python ../pipeline/uslr_linear/compute_graph.py --num_cores 2
    python ../pipeline/uslr_linear/compute_template.py
else
    echo "Running subject(s) " $@
    python ../pipeline/uslr_linear/initialize_graph.py --subjects $@
    python ../pipeline/uslr_linear/compute_graph.py --num_cores 2 --subjects $@
    python ../pipeline/uslr_linear/compute_template.py --subjects $@
fi
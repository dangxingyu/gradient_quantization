#!/bin/bash
export MASTER_ADDR=localhost
export MASTER_PORT=6000

torchrun --nproc_per_node=4 --nnode=1 utils.py
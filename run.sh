#!/bin/bash
export MASTER_ADDR=localhost
export MASTER_PORT=6000

CUDA_VISIBLE_DEVICES='0,1,2,3'
torchrun --nproc_per_node=2 --nnode=1 train.py
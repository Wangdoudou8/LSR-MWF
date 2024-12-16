#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0
exec -a python_cnndm_7_3_0.02_2e-3 python main.py > result/cnndm_7_3_0.02_2e-3.txt 2>&1 &
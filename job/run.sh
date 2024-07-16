#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:host=aurum-g01

docker exec -it -v $PWD:/workspace -w /workspace rkita-vqa bash -c "exec $SHELL -l && conda activate vqa-env && python main.py"

#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:host=aurum-g01

docker exec -it rkita-vqa bash -c "exec \$SHELL -l && conda activate vqa-env && python main.py" > log.txt

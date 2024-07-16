#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:host=aurum-g01

docker exec rkita-vqa /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/compat && source /opt/miniconda/bin/activate vqa-env && python main.py > log.log 2>&1"
#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:host=aurum-g01

docker exec rkita-vqa /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/compat:/opt/miniconda/envs/vqa-env/lib/python3.8/site-packages/nvidia/nvjitlink/lib && source /opt/miniconda/bin/activate vqa-env && python main.py > log2.log"
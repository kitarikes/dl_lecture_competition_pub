#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:host=aurum-g01

docker exec rkita-vqa /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat && source /opt/miniconda/bin/activate vqa-env && cd sandbox/dl_lecture_competition_pub && python main.py > log2.log"
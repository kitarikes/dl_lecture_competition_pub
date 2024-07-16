#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:host=aurum-g01

docker exec condescending_hermann /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/compat && source /opt/miniconda/bin/activate vqa-env && cd workspace/sandbox/dl_lecture_competition_pub && python main.py > log2.log"
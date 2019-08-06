#!/bin/bash

conda install conda=4.7.5
conda env create -f measureDLS.yml

conda activate measureDLS
pip install mnist
conda install keras
pip install opencv-python
conda deactivate

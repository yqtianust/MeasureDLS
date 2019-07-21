#!/bin/bash

conda install conda=4.7.5
conda env create -f measureDLS.yml

pip install mnist

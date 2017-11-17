#!/bin/bash
wget -O my_model.h5 'https://www.dropbox.com/s/vgtxzkvzb3ayauz/my_model.h5?dl=1'
python3 hw3pretrain.py $1 $2 my_model.h5



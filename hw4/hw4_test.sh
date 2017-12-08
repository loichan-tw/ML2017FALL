#!/bin/bash
#wget 'https://www.dropbox.com/s/vgtxzkvzb3ayauz/my_model.h5?dl=1'
python3 train.py model2.h5 test --test_path $1 --result_path $2 --load_model model2.h5



#!/usr/bin/env bash

### add PROTOKEN modules into python path
export PYTHONPATH=./PROTOKEN
python ./PROTOKEN/scripts/infer.py\
    --encoder_config ./PROTOKEN/config/encoder.yaml\
    --decoder_config ./PROTOKEN/config/decoder.yaml\
    --vq_config ./PROTOKEN/config/vq.yaml\
    --pdb_path ./example_scripts/results/example_pdbs/1opj.pdb\
    --save_dir_path ./example_scripts/results/example_pdbs/1opj\
    --load_ckpt_path ./ckpts/protoken_params_100000.pkl
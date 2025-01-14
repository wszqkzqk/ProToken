#!/usr/bin/env bash

### add PROTOKEN modules into python path
export PYTHONPATH=./PROTOKEN
python ./PROTOKEN/scripts/infer_batch.py\
    --encoder_config ./PROTOKEN/config/encoder.yaml\
    --decoder_config ./PROTOKEN/config/decoder.yaml\
    --vq_config ./PROTOKEN/config/vq.yaml\
    --pdb_dir_path ./example_scripts/results/example_pdbs\
    --save_dir_path ./example_scripts/results/example_pdbs/all\
    --load_ckpt_path ./ckpts/protoken_params_100000.pkl
#!/usr/bin/env bash

#SBATCH -o ./pdbs/result/encode_decode_structure_batch.out
#SBATCH -e ./pdbs/result/encode_decode_structure_batch.err
#SBATCH -p gpu31
#SBATCH -J zqk-encode-pdb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -t 200:00:00

# salloc -p gpu31 -J zqk2 --nodes=1 --gres=gpu:1 --ntasks-per-node=8 -t 200:00:00
### add PROTOKEN modules into python path
export PYTHONPATH=./PROTOKEN
mkdir -p ./pdbs/result
python ./PROTOKEN/scripts/infer_batch.py \
    --encoder_config ./PROTOKEN/config/encoder.yaml \
    --decoder_config ./PROTOKEN/config/decoder.yaml \
    --vq_config ./PROTOKEN/config/vq.yaml \
    --pdb_dir_path ./pdbs \
    --save_dir_path ./pdbs/result \
    --load_ckpt_path ./ckpts/protoken_params_100000.pkl

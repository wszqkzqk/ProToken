#!/usr/bin/env bash

#SBATCH -o ./pdbs/result/encode_decode_structure_batch.out
#SBATCH -e ./pdbs/result/encode_decode_structure_batch.err
#SBATCH -p gpu31
#SBATCH -J zqk-encode-pdb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -t 200:00:00

### add PROTOKEN modules into python path
export PYTHONPATH=./PROTOKEN
mkdir -p ./pdbs/result/decode_pdb
python ./PROTOKEN/scripts/decode_structure.py \
    --decoder_config ./PROTOKEN/config/decoder.yaml \
    --vq_config ./PROTOKEN/config/vq.yaml \
    --input_path ./pdbs/result/result_flatten.pkl \
    --output_dir ./pdbs/result/decode_pdb \
    --load_ckpt_path ./ckpts/protoken_params_100000.pkl \
    --padding_len 512

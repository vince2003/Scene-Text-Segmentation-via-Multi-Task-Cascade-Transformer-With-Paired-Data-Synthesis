export CUDA_VISIBLE_DEVICES=1

source activate lama

export TORCH_HOME=$(pwd) && export PYTHONPATH=.

python tools/train.py local_configs/segformer/B4/textseg.py 




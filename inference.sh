export CUDA_VISIBLE_DEVICES=1

source activate lama

export TORCH_HOME=$(pwd) && export PYTHONPATH=.


python3 bin/predict.py model.path=$(pwd)/experiments/models/ indir=$(pwd)/My_Input/ outdir=$(pwd)/SynImg_Resuts/


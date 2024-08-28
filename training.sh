export CUDA_VISIBLE_DEVICES=0

source activate lama

export TORCH_HOME=$(pwd) && export PYTHONPATH=.


# Run training
python3 bin/train.py -cn lama-fourier location=my_dataset data.batch_size=24


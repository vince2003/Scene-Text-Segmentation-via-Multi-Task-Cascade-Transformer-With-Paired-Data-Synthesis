export CUDA_VISIBLE_DEVICES=0

source activate lama

export TORCH_HOME=$(pwd) && export PYTHONPATH=.

# metrics calculation:
python3 bin/evaluate_predicts.py 


#---------------------------------------------------------
#$(pwd)/configs/eval2_gpu.yaml \
#$(pwd)/my_dataset/eval/random_thick_512/ \
#$(pwd)/inference/my_dataset/random_thick_512 \
#$(pwd)/inference/my_dataset/random_thick_512_metrics.csv

export CUDA_VISIBLE_DEVICES=1
source activate lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=.

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py\
 local_configs/segformer/B4/textseg.py\
  ./work_dirs/textseg/latest.pth\
   --eval mIoU\
    --eval-options efficient_test=True\
     --show-dir Textseg_results\


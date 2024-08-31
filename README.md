
# Scene Text Segmentation via Multi-Task Cascade Transformer With Paired Data Synthesis

This repository contains the code for scene text segmentation using a multi-task cascade transformer with paired data synthesis. The model is designed in two phases: **Paired Data Generation** and **Pixel-level Scene Text Segmentation**. This separation allows for effective data synthesis followed by robust text segmentation trained on a large volume of this synthetic data.

## Table of Contents
1. [Paired Data Generation](#paired-data-generation)
   - [Training](#training)
   - [Tracking Training](#tracking-training)
   - [Generating Data](#generating-data)
2. [Pixel-level Scene Text Segmentation](#pixel-level-scene-text-segmentation)
   - [Training](#training-pixel-level)
   - [Testing](#testing-pixel-level)
3. [Datasets](#datasets)
4. [Acknowledgments](#acknowledgments)
5. [Citation](#citation)

## I) Paired Data Generation

### Training

To generate paired data for training, follow these steps:

1. **Run the training script**:
   ```bash
   bash training.sh
   ```

2. **Execute the training command**:
   ```bash
   python3 bin/train.py -cn lama-fourier location=my_dataset data.batch_size=24
   ```

#### Parameters Explained

- **`-cn`**: Specifies the configuration model. This flag points to a YAML configuration file in the `configs/training` folder, allowing customization of model parameters. For instance, `"lama-fourier"` uses the configuration file `configs/training/lama-fourier.yaml`. You can modify this file to adjust parameters or select other models such as `"lama-regular.yaml"`, `"big-lama.yaml"`, or `"lama_small_train_masks.yaml"`.

- **`location=my_dataset data.batch_size=24`**: Directly updates the batch size to `24` for the dataset class `my_dataset` in the `lama-fourier.yaml` configuration file.

### Tracking Training

During training, you can monitor progress and model performance by accessing the `experiments` folder, which includes:

- **`config.yaml`**: Displays all the parameters of the model set for the current training session.
- **`models/`**: Stores model checkpoints, enabling training to resume from the last checkpoint or using the best-performing model for inference.
- **`samples/`**: Contains inference samples from the training, validation, and testing datasets generated during the training process. These samples provide direct evaluation of model performance.
- **`train.log`**: A log file that records the training process, including metrics like loss and accuracy over time.

### Generating Data

After training, generate paired data using the following steps:

1. **Run the inference script**:
   ```bash
   bash inference.sh
   ```

2. **Execute the prediction command**:
   ```bash
   python3 bin/predict.py model.path=$(pwd)/experiments/models/ indir=$(pwd)/My_Input/ outdir=$(pwd)/SynImg_Results/
   ```

#### Parameters Explained

- **`model.path`**: The directory containing the model checkpoint used for inference.
- **`indir`**: The input directory containing the data you want to process with the model.
- **`outdir`**: The output directory where the results of the model's predictions will be saved.

## II) Pixel-level Scene Text Segmentation

Our proposed text segmentation network is developed based on [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/), a toolbox that provides a framework for unified implementation and evaluation of semantic segmentation methods. It includes high-quality implementations of popular semantic segmentation methods and datasets.

### Training

To train the pixel-level scene text segmentation model, follow these steps:

1. **Change Directory to TextSegmentation**:
   ```bash
   cd TextSegmentation
   ```

2. **Run the training script**:
   ```bash
   bash training.sh
   ```

3. **Execute the training command**:
   ```bash
   python tools/train.py local_configs/segformer/B4/textseg.py
   ```

   - **`tools/train.py`**: Script for single-GPU training. For multi-GPU training, use `./tools/dist_train.sh`.
   - **`local_configs/segformer/B4/textseg.py`**: Configuration file for the model. Customize this file to adjust network architecture, dataset settings (augmentation, directory paths, etc.), training schedule, optimizer, and more.

### Testing

To test the model, follow these steps:

1. **Run the testing script**:
   ```bash
   bash testing.sh
   ```

2. **Execute the testing command**:
   ```bash
   python tools/test.py \
   local_configs/segformer/B4/textseg.py \
   ./work_dirs/textseg/latest.pth \
   --eval mIoU \
   --eval-options efficient_test=True \
   --show-dir Textseg_results
   ```

   - **`work_dirs`**: Directory containing the model checkpoint for testing.
   - **`--eval`**: Specifies the evaluation metrics to be used in the pixel-level text segmentation task.
   - **`--eval-options`**: Enhances memory efficiency during evaluation.
   - **`--show-dir`**: Directory where the results of the evaluation are saved.

## Datasets

We utilize several publicly available datasets for training and testing, including:

- **ICDAR13**
- **Total-Text**
- **TextSeg**
- **Coco-text**
- **ICDAR MLT 2017**
- **ICDAR MLT 2019**
- **ICDAR Art 2019**

## Acknowledgments

This project leverages code and resources from several repositories. We thank the authors and contributors of these projects for their valuable work:

- [CSAILVision Semantic Segmentation PyTorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
- [Lama](https://github.com/advimman/lama)
- [Open-MMLab](https://github.com/open-mmlab)
- [SegFormer](https://github.com/NVlabs/SegFormer)

## Citation

If you find this project useful in your research, please consider citing our work:

```bibtex
@article{dang2023scene,
  title={Scene text segmentation via multi-task cascade transformer with paired data synthesis},
  author={Dang, Quang-Vinh and Lee, Guee-Sang},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```

or

```bibtex
@inproceedings{dang2023scene,
  title={Scene text segmentation by paired data synthesis},
  author={Dang, Quang-Vinh and Lee, Guee-Sang},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={545--549},
  year={2023},
  organization={IEEE}
}
```

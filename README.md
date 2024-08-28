
# Scene Text Segmentation via Multi-Task Cascade Transformer With Paired Data Synthesis

## Training

To train the model, run the following commands:

```bash
bash training.sh
python3 bin/train.py -cn lama-fourier location=my_dataset data.batch_size=24
```

- **`cn`**: Specifies the configuration model. It accesses a file `*.yaml` in the `configs/training` folder. This folder contains many models that you can use to fit your requirements or customize the parameters of the available model. 

  For example, `"lama-fourier"` uses the config file `configs/training/lama-fourier.yaml`. You can set your own parameters by modifying this file. Furthermore, you can choose other models such as `"lama-regular.yaml"`, `"big-lama.yaml"`, `"lama_small_train_masks.yaml"`, etc.

- **`location=my_dataset data.batch_size=24`**: Updates the new value of `batch_size=24` directly in class `my_dataset` from `location` in the config file `lama-fourier.yaml`.

## Tracking Training

During training, you can track the progress by accessing the `experiments` folder, which includes:

- **`config.yaml`**: Shows all parameters of the model that you have already set.
- **`models`**: Saves checkpoints.
- **`samples`**: Contains inferences of the samples from the training, validation, and testing sets during the training process. These samples are used to directly evaluate how good or bad the model is.
- **`train.log`**: Shows the history of the training process.

## Testing

To test the model, run the following commands:

```bash
bash inference.sh
python3 bin/predict.py model.path=$(pwd)/experiments/models/ indir=$(pwd)/My_Input/ outdir=$(pwd)/SynImg_Results/
```

- **`model.path`**: The folder containing the checkpoint `best.ckp` for inference.
- **`indir`**: The input directory of data that you want to use as input for the model.
- **`outdir`**: Contains the results corresponding to the input.

## Dataset

We use the following available datasets: ICDAR13, Total-Text, TextSeg, Coco-text, ICDAR MLT 2017, ICDAR MLT 2019, and ICDAR Art 2019. The way to preprocess the data is provided as samples in the folder `My_Input`.

## Acknowledgments

This project makes use of code from the following repositories:

- [CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch)
- [Lama](https://github.com/advimman/lama?tab=readme-ov-file)
- [open-mmlab](https://github.com/open-mmlab)

## Citation

If you found this code helpful, please consider citing:

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

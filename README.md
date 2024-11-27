<div align='center'>

# From classification to segmentation with explainable AI: A study on crack detection and growth monitoring

[Florent Forest](https://florentfo.rest)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Hugo Porta](https://people.epfl.ch/hugo.porta)<sup>2</sup>&nbsp;&nbsp;&nbsp;
[Devis Tuia](https://people.epfl.ch/devis.tuia)<sup>2</sup>&nbsp;&nbsp;&nbsp;
[Olga Fink](https://people.epfl.ch/olga.fink)<sup>1</sup>
<br/>
<sub>
<sup>1</sup> Intelligent Maintenance and Operations Systems (IMOS), EPFL, Lausanne, Switzerland<br/>
<sup>2</sup> Environmental Computational Science and Earth Observation (ECEO), EPFL, Sion, Switzerland
</sub>

Automation in Construction, 2024

[![Paper](https://img.shields.io/badge/paper-AutCon-174c80)](https://www.sciencedirect.com/science/article/pii/S0926580524002334)&nbsp;&nbsp;&nbsp;[![Arxiv](https://img.shields.io/badge/arXiv-2309.11267-B31B1B)](https://arxiv.org/abs/2309.11267)

</div>

Source code for the implementation of the paper [From classification to segmentation with explainable AI: A study on crack detection and growth monitoring](https://www.sciencedirect.com/science/article/pii/S0926580524002334).

## Getting started

Install the dependencies:

```shell
pip install -r requirements.txt
```

### 1. Training the classifier

`MODEL` can be one of: `vgg11_128`, `vgg11`, `vgg11_bn_128`, `vgg11_bn`, `vgg11_128_bcos`, `vgg11_bcos`, `resnet18_deeplift`, `resnet34_deeplift`, `lrpvit`.

```shell
python class2seg/train.py \
    --model $MODEL \
    --num-classes 2 \
    --dataset DICCracksDataset \
    --data_path path/to/data/DIC_crack_dataset/ \
    --max_epochs 100 \
    --batch-size 64 \
    --output-dir output/my-output \
    [--pretrained] \ # initialize with ImageNet weights
    [--resume-from path/to/checkpoint.ckpt] \
    [--eval] \  # run evaluation only (requires resume from a checkpoint)
```

### 2. Extracting attribution maps

#### Most attribution maps

`METHOD` can be one of: `inputxgrad`, `intgrad`, `lrpVGG`/`lrpResNet`/`lrpViT`, `deeplift`, `gradientshap`, `deepliftshap`, `bcos_grad` (B-cos model only).

```shell
python class2seg/other_map.py \
    --cam_method $METHOD \
    --model_type $MODEL \
    --dataset DICCracksDataset \
    --data_path path/to/data/DIC_crack_dataset/ \
    --model_path path/to/checkpoint.ckpt \
    --output-dir path/to/output/ \
    --fold test \
    --batch_size 1 \
    [--baseline mean_baseline.pth] \ # for deeplift, deepliftshap and gradientshap only
    [--rules segmentation] # for LRP only
```

#### NN-Explainer (adapted for damage detection)

NN-Explainer uses config files in `class2seg/config_files` to set all parameters including the checkpoint of the trained classifier. First modify it to adapt it to your needs.

1. Train the explainer network

```shell
python class2seg/train_explainer.py -c class2seg/config_files/explainer_training/explainer_vgg11_128_unet11_dic_training.cfg
```

2. Extract explanations

```shell
python class2seg/train_explainer.py -c class2seg/config_files/testing_and_mask_saving/explainer_vgg11_128_unet11_dic_test_and_save_masks.cfg
```

#### U-Net supervised segmentation

```shell
python class2seg/unet_map.py \
    --model_type unet11 \
    --num-classes 1 \
    --dataset DICCracksDataset \
    --data_path path/to/data/DIC_crack_dataset/ \
    --model_path path/to/checkpoint.ckpt \
    --output-dir path/to/output/ \
    --fold test
```


### 3. Evaluate attribution maps

The post-processing method of the attribution maps `POST_METHOD` can be one of: `simple`, `gmm`, `simple+final`, `gmm+final`.

```shell
python class2seg/eval_seg.py \
    --dataset DICCracksDataset \
    --data_path path/to/data/DIC_crack_dataset/ \
    --image_path path/to/maps/ \
    --fold test \
    --post_method $POST_METHOD --model $MODEL
```

## Dataset

The DIC cracks dataset is available on [Zenodo](https://zenodo.org/records/4307686).

Additional negative images for classifier training will be made available.

## Citation

If our work was useful to you, please cite our paper:

```
@article{forest_classification_2024,
    title = {From classification to segmentation with explainable {AI}: {A} study on crack detection and growth monitoring},
    author = {Forest, Florent and Porta, Hugo and Tuia, Devis and Fink, Olga},
    journal = {Automation in Construction},
    year = {2024},
    volume = {165},
    issn = {0926-5805},
    pages = {105497},
    doi = {10.1016/j.autcon.2024.105497},
    url = {https://www.sciencedirect.com/science/article/pii/S0926580524002334},
}
```

## Acknowledgements

This work was supported by the EPFL ENAC Cluster grant “explAIn”. We also thank the [EPFL EESD](https://github.com/eesd-epfl) lab for providing additional images.

Thanks to all these great python packages or official implementations:
* Many XAI methods are implemented in [captum](https://github.com/pytorch/captum)
* Code for NN-Explainer borrowed from [NN-Explainer](https://github.com/stevenstalder/NN-Explainer)
* Code for B-cos networks borrowed from [B-cos](https://github.com/moboehle/B-cos/)
* Code for TernausNet borrowed from [TernausNet](https://github.com/ternaus/TernausNet)
* Code for LPRViT borrowed from [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability)

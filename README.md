# Blur Transformer

Research repository for **blur-based masked image modeling with Vision Transformers**.

This project extends the **SimMIM** framework by exploring alternative corruption mechanisms for masked image modeling, particularly **Gaussian blur-based corruption** instead of the standard **blank masking**.

The goal is to study how different corruption paradigms affect reconstruction learning and representation quality in Vision Transformers.

---

## Author

Akihito Maruya
Columbia University

---

## Based on

This repository builds on the official implementation of:

**SimMIM: A Simple Framework for Masked Image Modeling**
https://arxiv.org/abs/2111.09886

Original authors:

Zhenda Xie*, Zheng Zhang*, Yue Cao*, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai and Han Hu*

Original repository:
https://github.com/microsoft/SimMIM

---

# Overview

Masked Image Modeling (MIM) typically corrupts images by **removing patches (blank masking)** and training models to reconstruct them.

In this repository we explore an alternative approach:

Instead of removing information entirely, masked regions are replaced with **blurred versions of the image**. This allows us to investigate how models learn when the corruption corresponds to **loss of high-frequency information rather than missing pixels**.

Corruption types currently implemented include:

* **Blank masking** (standard SimMIM)
* **Gaussian blur masking**
* **Gaussian blur with frequency cutoff**

These corruption schemes allow controlled experiments on how Vision Transformers respond to different forms of information degradation.

---

# Repository Structure

```
Blur_transformer
│
├── configs/
│   Configuration files for pretraining and finetuning
│
├── models/
│   Vision Transformer implementations
│
├── main_simmim_blur.py
│   Pretraining script with blur corruption
│
├── main_finetune_blur.py
│   Finetuning for classification
│
├── simmim_helpers/
│   Evaluation utilities and analysis scripts
│
├── evaluation/
│   Reconstruction and noise experiments
│
└── figures/
```

---

# Pretraining

Example pretraining command:

```
python -m torch.distributed.run --standalone --nproc_per_node=4 main_simmim_blur.py \
  --cfg configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml \
  --data-path <imagenet-path>/train
```

Corruption types available:

```
blank
gauss_nocut
gauss_cutoff
```

---

# Finetuning

Example finetuning command:

```
python -m torch.distributed.run --standalone --nproc_per_node=4 main_finetune_blur.py \
  --cfg configs/vit_base__800ep/simmim_finetune__vit_base__img224__800ep.yaml \
  --data-path <imagenet-path> \
  --pretrained <checkpoint>
```

---

# Experiments

The repository supports experiments comparing:

### Corruption Type

* Blank masking
* Gaussian blur masking
* Gaussian blur with or without frequency cutoff (specify like dict(gaussian_sigma_ci=1.0, gaussian_cutoff_ci=3))

### Model Depth

* ViT depth 3
* ViT depth 6

### Mask Ratios

Typical values:

```
0.65
0.75
0.85
0.95
```

---

# Citation

If you use this repository, please cite the original SimMIM paper:

```
@inproceedings{xie2021simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhuliang and Dai, Qi and Hu, Han},
  booktitle={CVPR},
  year={2022}
}
```

---

# Acknowledgements

This work builds upon the official **SimMIM implementation** released by Microsoft Research.

https://github.com/microsoft/SimMIM


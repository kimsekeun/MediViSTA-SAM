# MediViSTA-SAM: Zero-shot Medical Video Analysis with Spatio-temporal SAM Adaptation


[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

This repo contains the code for our paper  <a href="https://arxiv.org/abs/2309.13539"> **MediViSTA-SAM: Zero-shot Medical Video Analysis with Spatio-temporal SAM Adaptation**  </a>.

![Overview of framework](method.png?raw=true "Overview of MeediViSTA framework")

> The Segmentation Anything Model (SAM) has attracted considerable attention as a foundational model well-known for its robust generalization capabilities across various downstream tasks. However, SAM does not exhibit satisfactory performance in the realm of medical image analysis. In this study, we introduce the first study on adapting SAM on video segmentation, called MediViSTA-SAM, a novel approach designed for medical video segmentation. Given video data, MediViSTA, spatio-temporal adapter captures long and short range temporal attention with cross-frame attention mechanism effectively constraining it to consider the immediately preceding video frame as a reference, while also considering spatial information effectively. Additionally, it incorporates multi-scale fusion by employing a U-shaped encoder and a modified mask decoder to handle objects of varying sizes. To evaluate our approach, extensive experiments were conducted using state-of-the-art (SOTA) methods, assessing its generalization abilities on multi-vendor in-house echocardiography datasets. The results highlight the accuracy and effectiveness of our network in medical video segmentation.


## Execution Instructions
- Envrionment Setting

```
pip install -r requirements.py
```
  
- Build Model
```
from models.segmentation.segment_anything import sam_model_registry
model, img_embedding_size = sam_model_registry[args.vit_type](args, image_size=args.img_size,
                                                num_classes=args.num_classes,
                                                chunk = chunk,
                                                checkpoint=args.resume, pixel_mean=[0., 0., 0.],
                                                pixel_std=[1., 1., 1.])
```

## Pretrained Model Chcekpoints
We employed pretrained SAM model to train our model. 
Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Results

- Qualitative results
![Experimental results](results.png?raw=true "Experimental results")

- Zero-shot analysis on in-house MGH dataset
![Experimental results](results_.png?raw=true "Experimental results")


## Citation

If you found MediViSTA-SAM useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@article{kim2023medivista,
  title={MediViSTA-SAM: Zero-shot Medical Video Analysis with Spatio-temporal SAM Adaptation},
  author={Kim, Sekeun and Kim, Kyungsang and Hu, Jiang and Chen, Cheng and Lyu, Zhiliang and Hui, Ren and Kim, Sunghwan and Liu, Zhengliang and Zhong, Aoxiao and Li, Xiang and others},
  journal={arXiv preprint arXiv:2309.13539},
  year={2023}
}
```

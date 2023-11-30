# MediViSTA-SAM: Zero-shot Medical Video Analysis with Spatio-temporal SAM Adaptation


[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

This repo contains the code for our paper **MediViSTA-SAM: Zero-shot Medical Video Analysis with Spatio-temporal SAM Adaptation**.
We are in underconstruction this page.

## Execution Instructions
- Envrionment
  Docker link -> https://hub.docker.com/repository/docker/sk1064/medivista/general
  
- Build Model
```
from models.segmentation.segment_anything import sam_model_registry
model, img_embedding_size = sam_model_registry[args.vit_type](args, image_size=args.img_size,
                                                num_classes=args.num_classes,
                                                chunk = chunk,
                                                checkpoint=args.resume, pixel_mean=[0., 0., 0.],
                                                pixel_std=[1., 1., 1.])
```
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

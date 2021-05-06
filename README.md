# <p align=center>`awesome Transformer in CV papers`</p>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A collection of resources on Transformer in CV.

## <span id="head-content"> *Content* </span>
* - [ ] [1. Description](#head1)
* - [ ] [2. Survey](#head2)
* - [ ] [3. Paper With Code](#head2)
  * - [ ] [Image Classification](#head-Image-Classification)
  * - [ ] [Object Detection](#head-Object-Detection)
  * - [ ] [Instance Segmentation](#head-Instance-Segmentation)
  * - [ ] [Semantic Segmentation](#head-Semantic-Segmentation)
  * - [ ] [Video Understanding](#head-Video-Understanding)
  * - [ ] [Monocular Depth Estimation](#head-Monocular-Depth-Estimation)
  * - [ ] [GAN](#head-GAN)
  * - [ ] [Deepfake Detection](#head-Deepfake-Detection) 
  * - [ ] [Perceptual Representation](#head-Perceptual-Representation)
* [*Contact Me*](#head3)

## <span id="head1"> *1. Description* </span>
* ⭐: State-of-the-Art
* 🌱: Novel idea
* 📌: The first...
* 👑: Novel dataset

## <span id="head2"> *2. Survey* </span>
* 『[**Visual Transformer Blog**](https://blog.csdn.net/u014636245/article/details/116333223) in Chinese』Proudly produced by [@JieJi](https://blog.csdn.net/u014636245)
* (arXiv preprint 2021) **Transformers in Vision: A Survey** [[v1](https://arxiv.org/pdf/2101.01169v1.pdf)](2021.01.04) [[v2](https://arxiv.org/pdf/2101.01169.pdf)](2021.02.22)
* (arXiv preprint 2020+2021) **A Survey on Visual Transformer** [[v1](https://arxiv.org/pdf/2012.12556v1.pdf)](2020.12.23) [[v2](https://arxiv.org/pdf/2012.12556v2.pdf)](2021.01.15) [[v3](https://arxiv.org/pdf/2012.12556v3.pdf)](2021.01.30)
## <span id="head3"> *3. Paper With Code* </span>
  * <span id="head-Image-Classification"> **Image Classification**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/microsoft/Swin-Transformer)] 
  * <span id="head-Object-Detection"> **Object Detection**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)] ⭐SOTA on [COCO test-dev, COCO minival](https://cocodataset.org/#home)
      * (arXiv preprint 2021) **Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**, Wenhai Wang et al. [[Paper](https://arxiv.org/pdf/2102.12122v1.pdf)] [[Code](https://github.com/whai362/PVT)] 
  * <span id="head-Instance-Segmentation"> **Instance Segmentation**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)] ⭐SOTA on [COCO test-dev, COCO minival](https://cocodataset.org/#home)  
  * <span id="head-Semantic-Segmentation"> **Semantic Segmentation**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)] ⭐SOTA on [ADE20K dataset, ADE20K val](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
      * (arXiv preprint 2021) **Vision Transformers for Dense Prediction**, René Ranftl et al. [[Paper](https://arxiv.org/pdf/2103.13413.pdf)] [[Code](https://github.com/intel-isl/DPT)]
 * <span id="head-Image-Retrieval"> **Image Retrieval**  </span>
      * (arXiv preprint 2021) **TransHash: Transformer-based Hamming Hashing for Efficient Image Retrieval**, Yongbiao Chen et al. [[Paper](https://arxiv.org/pdf/2105.01823.pdf)] [[Code](Todo)] 🌱the first work to tackle deep hashing learning problems without convolutional neural networks.
 * <span id="head-Video-Understanding"> **Video Understanding**  </span>
      * (arXiv preprint 2021) **Is Space-Time Attention All You Need for Video Understanding?**, Gedas Bertasius et al. [[Paper](https://arxiv.org/pdf/2102.05095.pdf)] [[Code](https://github.com/facebookresearch/TimeSformer)] 🌱Convolution-free  
 * <span id="head-Monocular-Depth-Estimation"> **Monocular Depth Estimation**  </span>
      * (arXiv preprint 2021) **Vision Transformers for Dense Prediction**, René Ranftl et al. [[Paper](https://arxiv.org/pdf/2103.13413.pdf)] [[Code](https://github.com/intel-isl/DPT)] ⭐SOTA on [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
 * <span id="head-GAN"> **GAN**  </span>
      * (arXiv preprint 2021) **TransGAN: Two Transformers Can Make One Strong GAN**, Yifan Jiang et al. [[Paper](https://arxiv.org/pdf/2102.07074.pdf)] [[Code](https://github.com/VITA-Group/TransGAN)] 🌱Convolution-free
  * <span id="head-Deepfake-Detection"> **Deepfake Detection**  </span>
      * (arXiv preprint 2021) **M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection**, Junke Wang et al. [[Paper](https://arxiv.org/pdf/2104.09770v2.pdf)] 📌The first Multi-modal Multi-scale Transformer 👑face **S**wapping and facial **R**eenactment **D**eep**F**ake(SR-DF) Dataset
  * <span id="head-Perceptual-Representation"> **Perceptual Representation** </span>
      * (CVPR 2021-[NTIRE workshop](https://data.vision.ee.ethz.ch/cvl/ntire21/)) **Perceptual Image Quality Assessment with Transformers**, Manri Cheon et al. [[Paper](https://arxiv.org/pdf/2104.14730.pdf)] [[Code](https://github.com/manricheon/IQT)] ⭐1st Place in [NTIRE 2021 perceptual IQA challenge](https://competitions.codalab.org/competitions/28050#learn_the_details). 
  * <span id="head-Other-Applications"> **Image Retrieval**  </span>
      * (arXiv preprint 2021) **Novelty Detection and Analysis of Traffic Scenario Infrastructures in  the Latent Space of a Vision Transformer-Based Triplet Autoencoder**, Jonas Wurst et al. [[Paper](https://arxiv.org/pdf/2105.01924.pdf)] [[Code](https://github.com/JWTHI/ViTAL-SCENE)] 🌱 Triplet Training
## <span id="head3"> *Contact Me* </span>

* [Yutong ZHOU](https://github.com/Yutong-Zhou-cv) in [Interaction Laboratory, Ritsumeikan University.](https://github.com/Rits-Interaction-Laboratory) (づ￣0￣)づ

* If you have any question, please feel free to contact Yutong ZHOU (E-mail: <zhou@i.ci.ritsumei.ac.jp>).

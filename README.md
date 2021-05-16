# <p align=center>`awesome Transformer in CV papers`</p>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A collection of resources on Transformer in CV.

## <span id="head-content"> *Content* </span>
* - [ ] [1. Description](#head1)
* - [ ] [2. Survey](#head2)
* - [ ] [3. Paper With Code](#head3)
  * - [ ] [Vision and Language](#head-Vision-and-Language)
  * - [ ] [Image Classification](#head-Image-Classification)
  * - [ ] [Object Detection](#head-Object-Detection)
  * - [ ] [Object Tracking](#head-Object-Tracking)
  * - [ ] [Instance Segmentation](#head-Instance-Segmentation)
  * - [ ] [Semantic Segmentation](#head-Semantic-Segmentation)
  * - [ ] [Image Retrieval](#head-Image-Retrieval)
  * - [ ] [Video Understanding](#head-Video-Understanding)
  * - [ ] [Monocular Depth Estimation](#head-Monocular-Depth-Estimation)
  * - [ ] [GAN](#head-GAN)
  * - [ ] [Deepfake Detection](#head-Deepfake-Detection) 
  * - [ ] [Perceptual Representation](#head-Perceptual-Representation)
  * - [ ] [Low Level Vision](#head-Low-Level-Vision)
  * - [ ] [Sign Language](#head-Sign-Language)
  * - [ ] [Other Applications](#head-Other-Applications)
  * - [ ] [Byond Transformer](#head-Byond-Transformer)
* [*Contact Me*](#head4)

## <span id="head1"> *1. Description* </span>
* ```🌱: Novel idea```
* ```📌: The first...```
* ⭐: State-of-the-Art
* 👑: Novel dataset

## <span id="head2"> *2. Survey* </span>
* 『[**Visual Transformer Blog**](https://blog.csdn.net/u014636245/article/details/116333223) in Chinese』Proudly produced by [@JieJi](https://blog.csdn.net/u014636245)
* (arXiv preprint 2021) **Transformers in Vision: A Survey** [[v1](https://arxiv.org/pdf/2101.01169v1.pdf)](2021.01.04) [[v2](https://arxiv.org/pdf/2101.01169.pdf)](2021.02.22)
* (arXiv preprint 2020+2021) **A Survey on Visual Transformer** [[v1](https://arxiv.org/pdf/2012.12556v1.pdf)](2020.12.23) [[v2](https://arxiv.org/pdf/2012.12556v2.pdf)](2021.01.15) [[v3](https://arxiv.org/pdf/2012.12556v3.pdf)](2021.01.30)
## <span id="head3"> *3. Paper With Code* </span>

  * <span id="head-Vision-and-Language"> **Vision and Language**  </span>
      * (arXiv preprint 2021) **Episodic Transformer for Vision-and-Language Navigation**, Alexander Pashevich et al. [[Paper](https://arxiv.org/pdf/2105.06453.pdf)] [[Code](https://github.com/alexpashevich/E.T.)] 
        * ```🌱 An attention-based architecture for vision-and-language navigation. ```
        * ```🌱 Use synthetic instructions as the intermediate interface between the human and the agent. ```
        * ⭐ SOTA on [ALFRED](https://github.com/askforalfred/alfred) 
      * (CVPR 2021 [AI for Content Creation Workshop](http://visual.cs.brown.edu/workshops/aicc2021/)) **High-Resolution Complex Scene Synthesis with Transformers**, Manuel Jahn et al. [[Paper](https://arxiv.org/pdf/2105.06458.pdf)] 
        * ```🌱 An orthogonal approach to the controllable synthesis of complex scene images, where the generative model is based on pure likelihood training without additional objectives. ```
      * (ICML 2021) **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**, Wonjae Kim et al. [[Paper](https://arxiv.org/pdf/2102.03334.pdf)] [[Code](https://github.com/dandelin/vilt)] 
        * ```🌱 Without region features or deep convolutional visual encoders. ```
        * ```🌱 Drive performance on whole word masking and image augmentations in Vision-and-Language Pretraining (VLP) training schemes. ```
      * (arXiv preprint 2021) **VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning**, Jun Chen et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/Vision-CAIR/VisualGPT)] 
        * ```📌 The first large pretrained language models for image captioning. ```
      * (CVPR 2021) **Kaleido-BERT: Vision-Language Pre-training on Fashion Domain**, Mingchen Zhuge et al. [[Paper](https://arxiv.org/pdf/2103.16110.pdf)] [[Code](https://github.com/mczhuge/Kaleido-BERT/)] [[Video](http://dpfan.net/wp-content/uploads/Kaleido-BERT.mp4)] 
        * ```📌 The first method extracts a series of multi-grained image patches for the image modality.``` 
        * ```🌱 Kaleido strategy ```
      * (arXiv preprint 2020) **ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data**, Di Qi et al. [[Paper](https://arxiv.org/pdf/2001.07966.pdf)] 
        * ```🌱 Pre-training with a large-scale Image-Text dataset. ```
      * (CVPR 2020) **12-in-1: Multi-Task Vision and Language Representation Learning**, Jiasen Lu et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.pdf)] [[Code](https://github.com/facebookresearch/vilbert-multi-task)]  
        * ```🌱 Multi-task training ```
      * (NeurlIPS 2019) **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks**, Jiasen Lu et al. [[Paper](https://arxiv.org/pdf/1908.02265.pdf)] [[Code](https://github.com/facebookresearch/vilbert-multi-task)] 
        * ```🌱 Cross-modality co-attention layers. ```
      * (ICCV 2019) **VideoBERT: A Joint Model for Video and Language Representation Learning**, Chen Sun et al. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_VideoBERT_A_Joint_Model_for_Video_and_Language_Representation_Learning_ICCV_2019_paper.pdf)] [[Code](https://github.com/ammesatyajit/VideoBERT)] 
        * ```📌 The first Video-Text Pre-Training Model. ```
      * (arXiv preprint 2019) **VisualBERT: A Simple and Performant Baseline for Vision and Language**, Liunian Harold Li et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/uclanlp/visualbert)] 
        * ```📌 The first Image-Text Pre-Training Model. ```
      * (arXiv preprint 2019) **Visual Grounding with Transformers**, Ye Du et al. [[Paper](https://arxiv.org/pdf/2105.04281.pdf)] [[Code](https://github.com/uclanlp/visualbert)] 
        * ```🌱 Visual grounding task. ```
 
  * <span id="head-Image-Classification"> **Image Classification**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/microsoft/Swin-Transformer)] 
      * (arXiv preprint 2021) **Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet**, Luke Melas-Kyriazi [[Paper](https://arxiv.org/pdf/2105.02723.pdf)] [[Code](https://github.com/lukemelas/do-you-even-need-attention)] 
        * ```🌱 Attention Layer-free ```
      * (arXiv preprint 2021) **Conformer: Local Features Coupling Global Representations for Visual Recognition**, Zhiliang Peng et al[[Paper](https://arxiv.org/pdf/2105.03889.pdf)] [[Code](https://github.com/pengzhiliang/Conformer)] 
        * ```🌱 A hybrid network structure with Convs and attention mechanisms.```
      * (arXiv preprint 2021) **Self-Supervised Learning with Swin Transformers**, Zhenda Xie et al [[Paper](https://arxiv.org/pdf/2105.04553.pdf)] [[Code](https://github.com/SwinTransformer/Transformer-SSL)] 
        * ```🌱 A self-supervised learning approach based on vision transformers as backbone.```
      * (arXiv preprint 2021) **Twins: Revisiting the Design of Spatial Attention in Vision Transformers**, Xiangxiang Chu et al [[Paper](https://arxiv.org/pdf/2104.13840.pdf)] [[Code](https://github.com/Meituan-AutoML/Twins)] 
        * ```🌱 Two vision transformer architectures(Twins- PCPVT and Twins-SVT). ```
        * ```🌱 May serve as stronger backbones for many vision tasks.```

  * <span id="head-Object-Detection"> **Object Detection**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)] 
        * ⭐ SOTA on [COCO test-dev, COCO minival](https://cocodataset.org/#home)
      * (arXiv preprint 2021) **Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**, Wenhai Wang et al. [[Paper](https://arxiv.org/pdf/2102.12122v1.pdf)] [[Code](https://github.com/whai362/PVT)] 
      * (arXiv preprint 2021) **Twins: Revisiting the Design of Spatial Attention in Vision Transformers**, Xiangxiang Chu et al [[Paper](https://arxiv.org/pdf/2104.13840.pdf)] 

  * <span id="head-Object-Tracking"> **Object Tracking**  </span>
      * (arXiv preprint 2021) **MOTR: End-to-End Multiple-Object Tracking with TRansformer**, Fangao Zeng et al. [[Paper](https://arxiv.org/pdf/2105.03247.pdf)] [[Code](https://github.com/megvii-model/MOTR)] 
        * ```📌 The first fully end-toend multiple-object tracking framework. ```
        * ```🌱 Model the long-range temporal variation of the objects. ```
        * ```🌱 Introduce the concept of “track query” to models the entire track of an object.```
      * (arXiv preprint 2021) **TrTr: Visual Tracking with Transformer**, Moju Zhao et al. [[Paper](https://arxiv.org/pdf/2105.03817.pdf)] [[Code](https://github.com/tongtybj/TrTr)] 
        * ```🌱 Transformer models template and search in targe image.```
    
  * <span id="head-Instance-Segmentation"> **Instance Segmentation**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)] 
        * ⭐ SOTA on [COCO test-dev, COCO minival](https://cocodataset.org/#home) 
     
  * <span id="head-Semantic-Segmentation"> **Semantic Segmentation**  </span>
      * (arXiv preprint 2021) **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, Ze Liu et al. [[Paper](https://arxiv.org/pdf/2103.14030.pdf)] [[Code](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)] 
        * ⭐ SOTA on [ADE20K dataset, ADE20K val](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
      * (arXiv preprint 2021) **Vision Transformers for Dense Prediction**, René Ranftl et al. [[Paper](https://arxiv.org/pdf/2103.13413.pdf)] [[Code](https://github.com/intel-isl/DPT)]
      * (arXiv preprint 2021) **Twins: Revisiting the Design of Spatial Attention in Vision Transformers**, Xiangxiang Chu et al [[Paper](https://arxiv.org/pdf/2104.13840.pdf)] 
      * (arXiv preprint 2021) **Segmenter: Transformer for Semantic Segmentation**, Robin Strudel et al. [[Paper](https://arxiv.org/pdf/2105.05633.pdf)] [[Code](https://github.com/rstrudel/segmenter)] 
         * ```🌱 Convolution-free ```
         * ```🌱 Capture contextual information by design and outperform Fully Convolutional Networks(FCN) based approaches.```

  * <span id="head-Image-Retrieval"> **Image Retrieval**  </span>
      * (arXiv preprint 2021) **TransHash: Transformer-based Hamming Hashing for Efficient Image Retrieval**, Yongbiao Chen et al. [[Paper](https://arxiv.org/pdf/2105.01823.pdf)] [[Code](Todo)] 
        * ```📌 The first work to tackle deep hashing learning problems without convolutional neural networks. ```
        * ```🌱 Convolution-free ```
 * <span id="head-Video-Understanding"> **Video Understanding**  </span>
      * (arXiv preprint 2021) **ViViT: A Video Vision Transformer**, Anurag Arnab et al. [[Paper](https://arxiv.org/pdf/2103.15691v1.pdf)] [[Code](https://github.com/rishikksh20/ViViT-pytorch)] 
        * ```🌱 Convolution-free ``` 
        * ⭐ SOTA on [Kinetics-400/600](https://deepmind.com/research/open-source/kinetics)
      * (arXiv preprint 2021) **Is Space-Time Attention All You Need for Video Understanding?**, Gedas Bertasius et al. [[Paper](https://arxiv.org/pdf/2102.05095.pdf)] [[Code](https://github.com/facebookresearch/TimeSformer)] 
        * ```🌱 Convolution-free ``` 
     
 * <span id="head-Monocular-Depth-Estimation"> **Monocular Depth Estimation**  </span>
      * (arXiv preprint 2021) **Vision Transformers for Dense Prediction**, René Ranftl et al. [[Paper](https://arxiv.org/pdf/2103.13413.pdf)] [[Code](https://github.com/intel-isl/DPT)] 
        * ⭐ SOTA on [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
     
 * <span id="head-GAN"> **GAN**  </span>
      * (arXiv preprint 2021) **TransGAN: Two Transformers Can Make One Strong GAN**, Yifan Jiang et al. [[Paper](https://arxiv.org/pdf/2102.07074.pdf)] [[Code](https://github.com/VITA-Group/TransGAN)] 
        * ```🌱 Convolution-free```
     
  * <span id="head-Deepfake-Detection"> **Deepfake Detection**  </span>
      * (arXiv preprint 2021) **M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection**, Junke Wang et al. [[Paper](https://arxiv.org/pdf/2104.09770v2.pdf)] 
        * ```📌 The first Multi-modal Multi-scale Transformer ```
        * 👑 face **S**wapping and facial **R**eenactment **D**eep**F**ake(SR-DF) Dataset
     
  * <span id="head-Perceptual-Representation"> **Perceptual Representation** </span>
      * (CVPR 2021-[NTIRE workshop](https://data.vision.ee.ethz.ch/cvl/ntire21/)) **Perceptual Image Quality Assessment with Transformers**, Manri Cheon et al. [[Paper](https://arxiv.org/pdf/2104.14730.pdf)] [[Code](https://github.com/manricheon/IQT)] 
        * ⭐ 1st Place in [NTIRE 2021 perceptual IQA challenge](https://competitions.codalab.org/competitions/28050#learn_the_details).
     
  * <span id="head-Low-Level-Vision"> **Low Level Vision**  </span>
      * (arXiv preprint 2021) **Pre-Trained Image Processing Transformer**, Hanting Chen et al. [[Paper](https://arxiv.org/pdf/2012.00364v2.pdf)] [[Code](https://github.com/huawei-noah/Pretrained-IPT)] [[2nd code](https://github.com/perseveranceLX/ImageProcessingTransformer)] 
        * ```🌱 Various image processing tasks based Transformer.```
     
  * <span id="head-Sign-Language"> **Sign Language**  </span>
      * (arXiv preprint 2021) **Aligning Subtitles in Sign Language Videos**, Hannah Bull et al.  [[Paper](https://arxiv.org/pdf/2105.02877.pdf)] [[Project](https://www.robots.ox.ac.uk/~vgg/research/bslalign/)] 
        * ```📌 The first subtitle alignment task based on Transformers.```
      * (arXiv preprint 2021) **Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and Mixture Density Networks**, Ben Saunders et al.  [[Paper](https://arxiv.org/pdf/2103.06982.pdf)]  
        * ```📌 The first Sign Language Production(SLP) model to translate from discrete spoken language sentences to continuous 3D multi-channel sign pose sequences in an end-to-end manner. ```
        * (Extended journal version of **Progressive Transformers for End-to-End Sign Language Production**)
      * (WACV 2021) **Pose-based Sign Language Recognition using GCN and BERT**, Anirudh Tunga et al.  [[Paper](https://openaccess.thecvf.com/content/WACV2021W/HBU/papers/Tunga_Pose-Based_Sign_Language_Recognition_Using_GCN_and_BERT_WACVW_2021_paper.pdf)] 
        * ```🌱 Capture the spatial interactions in every frame comprehensively before utilizing the temporal dependencies between various frames.```
      * (CVPR 2020) **Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation**, Necati Cihan Camgoz et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.pdf)] [[Code](https://github.com/neccam/slt)] 
        * ```📌 The first successful application of transformers for Continuous Sign Language Recognition(CSLR) and Sign Language Translation(SLT). ```
        * ```🌱 A novel multi-task formalization of CSLR and SLT exploits the supervision power of glosses, without limiting the translation to spoken language.```
      * (COLING 2020 & ECCV 2020 SLRTP Workshop) **Better Sign Language Translation with STMC-Transformer**, Kayo Yin et al. [[Paper](https://arxiv.org/pdf/2004.00588v2.pdf)] [[Code](https://github.com/kayoyin/transformer-slt)] 
        * ```📌 The first work adopts weight tying, transfer learning, and ensemble learning in Sign Language Translation(SLT). ```
        * ⭐ SOTA on [ASLG-PC12](https://achrafothman.net/site/english-asl-gloss-parallel-corpus-2012-aslg-pc12/)
      * (ECCV 2020 Workshop) **Multi-channel Transformers for Multi-articulatory Sign Language Translation**, Necati Cihan Camgoz et al.  [[Paper](https://link.springer.com/content/pdf/10.1007%2F978-3-030-66823-5_18.pdf)] 
        * ```📌 The first successful approach to multi-articulatory Sign Language Translation(SLT), which models the inter and intra contextual relationship of manual and non-manual channels. ```
        * ```🌱 A novel multi-channel transformer architecture supports multi-channel, asynchronous, sequence-to-sequence learning.```
      * (ECCV 2020) **Progressive Transformers for End-to-End Sign Language Production**, Ben Saunders et al. [[Paper](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58621-8_40.pdf)] [[Code](https://github.com/BenSaunders27/ProgressiveTransformersSLP)]
      * (ECCV 2020) **Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition**, Zhe Niu et al. [[Paper](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58517-4_11.pdf)] [[Code](https://github.com/zheniu/stochastic-cslr)] 
        * ```🌱 Propose stochastic frame dropping (SFD) and stochastic gradient stopping (SGS) to reduce video memory footprint, improve model robustness and alleviate the overfitting problem during model training.```
    
  * <span id="head-Other-Applications"> **Other Applications**  </span>
      * (CVPR 2021) [Human Pose and Mesh Reconstruction ] **End-to-End Human Pose and Mesh Reconstruction with Transformers**, Kevin Lin et al. [[Paper](https://arxiv.org/pdf/2012.09760.pdf)]
        * ```📌 The first approach leverages a transformer encoder architecture to learn 3D human pose and mesh reconstruction from a single input image. ```
        * ```🌱 Able to predict a different type of 3D mesh, such as 3D hand. ```
        * ⭐ SOTA on [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [3DPW](http://virtualhumans.mpi-inf.mpg.de/3DPW/)
      * (arXiv preprint 2021) [Traffic Scenario Infrastructures] **Novelty Detection and Analysis of Traffic Scenario Infrastructures in  the Latent Space of a Vision Transformer-Based Triplet Autoencoder**, Jonas Wurst et al. [[Paper](https://arxiv.org/pdf/2105.01924.pdf)] [[Code](https://github.com/JWTHI/ViTAL-SCENE)] 
        * ```🌱 Triplet Training```
      * (arXiv preprint 2021) [Handwritten Recognition] **Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer**, Wenqi Zhao et al. [[Paper](https://arxiv.org/pdf/2105.02412.pdf)] [[Code](https://github.com/Green-Wood/BTTR)] 
        * ```🌱 Handwritten Mathematical Expression Recognition```
      * (arXiv preprint 2021) [Scene Flow Estimation] **SCTN: Sparse Convolution-Transformer Network for Scene Flow Estimation**, Bing Li et al. [[Paper](https://arxiv.org/pdf/2105.04447.pdf)] 
        * ```📌 The first Sparse Convolution-Transformer Network (SCTN) for scene flow estimation.```
      * (arXiv preprint 2021) [Medical Image Segmentation] **Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation**, Hu Cao et al. [[Paper](https://arxiv.org/pdf/2105.05537.pdf)]  [[Code](https://github.com/HuCaoFighting/Swin-Unet)] 
        * ```📌 The first pure Transformer-based U-shaped architecture .```
      * (arXiv preprint 2021) [Image Registration] **Attention for Image Registration (AiR): an unsupervised Transformer approach**, Zihao Wang et al. [[Paper](https://arxiv.org/pdf/2105.02282.pdf)]
        * ```📌 The first Transformer based image unsupervised registration method.```
        * ```🌱 A multi-scale attention parallel Transformer framework.```
      * (arXiv preprint 2021) [Action Recognition] **VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text**, Hassan Akbari et al. [[Paper](https://arxiv.org/pdf/2104.11178.pdf)]
        * ```🌱 Convolution-free```
        * ```🌱 Multimodal representation```
        * ⭐ SOTA on [Moments in Time](http://moments.csail.mit.edu/)
      * (arXiv preprint 2021) [Action Recognition] **An Image is Worth 16x16 Words, What is a Video Worth?**, Gilad Sharir et al. [[Paper](https://arxiv.org/pdf/2103.13915.pdf)]  [[Code](https://github.com/Alibaba-MIIL/STAM)] 
        * ```🌱 Achieves 78.8 top1-accuracy with ×40 faster inference time on Kinetics-400 benchmark.```
        * ```🌱 End-to-end trainable ```
      * (arXiv preprint 2021) [Video Prediction] **Local Frequency Domain Transformer Networks for Video Prediction**, Hafez Farazi et al. [[Paper](https://arxiv.org/ftp/arxiv/papers/2105/2105.04637.pdf)]  [[Code](https://github.com/AIS-Bonn/Local_Freq_Transformer_Net)] 
        * ```📌 The first pure Transformer-based U-shaped architecture .```
        * ```🌱 Lightweight and flexible, enabling use as a building block at the core of sophisticated video prediction systems. ```
 
  * <span id="head-Byond-Transformer"> **Byond Transformer**  </span>
      * (arXiv preprint 2021) **MLP-Mixer: An all-MLP Architecture for Vision**, Ilya Tolstikhin et al. [[Paper](https://arxiv.org/pdf/2105.01601v1.pdf)] [[Code](https://github.com/google-research/vision_transformer)] 
        * ```🌱 An architecture based exclusively on multi-layer perceptrons (MLPs). ```
      * (arXiv preprint 2021) **Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks**, Meng-Hao Guo et al. [[Paper](https://arxiv.org/pdf/2105.02358.pdf)] [[Code](https://github.com/MenghaoGuo/-EANet)] 
        * ```🌱 Simply using two cascaded linear layers and two normalization layers.```
      * (arXiv preprint 2021) **Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet**, Luke Melas-Kyriazi. [[Paper](https://arxiv.org/abs/2105.02723)] [[Code](https://github.com/lukemelas/do-you-even-need-attention)] 
        * ```🌱 Replace the attention layer in a vision transformer with a feed-forward layer.```
      * (arXiv preprint 2021) **RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition**, Xiaohan Ding et al. [[Paper](https://arxiv.org/pdf/2105.01883.pdf)] [[Code](https://github.com/DingXiaoH/RepMLP)] 
        * ```🌱 Re-parameterizing Convolutions and MLP```
      * (arXiv preprint 2021) **ResMLP: Feedforward networks for image classification with data-efficient training**, Hugo Touvron et al. [[Paper](https://arxiv.org/pdf/2105.03404.pdf)] [[Code](https://github.com/lucidrains/res-mlp-pytorch)] 
        * ```🌱 Residual MLP structure ```

<!--#comments * (arXiv preprint 2021) **Title**, first author et al. [[Paper]()] [[Code]()] 🌱 tips-->

## <span id="head4"> *Contact Me* </span>

* [Yutong ZHOU](https://github.com/Yutong-Zhou-cv) in [Interaction Laboratory, Ritsumeikan University.](https://github.com/Rits-Interaction-Laboratory) (づ￣0￣)づ

* If you have any question, please feel free to contact Yutong ZHOU (E-mail: <zhou@i.ci.ritsumei.ac.jp>).

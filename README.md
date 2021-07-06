# Separating Content and Style in Unsupervised Image-to-Image Translation

![Python 3.7](https://img.shields.io/badge/python-3.7-DodgerBlue.svg?style=plastic)
![Pytorch 1.30](https://img.shields.io/badge/pytorch-1.3.0-DodgerBlue.svg?style=plastic)
![CUDA 10.1](https://img.shields.io/badge/cuda-10.1-DodgerBlue.svg?style=plastic)


> **Separating Content and Style in Unsupervised Image-to-Image Translation**
>
> **Abstract:**  Unsupervised image-to-image translation aims to learn the mapping between two visual domains with unpaired samples. The existing works often learn the domain-invariant content code and domain-specific style code individually using two encoders. However, this compromises the content and style representation, makes the content code focuses on trivial regions that are shared between domains (*e.g.*, background) and the style code focuses only on the global appearance. In this paper, we propose to extract and separate content code and style code within a single encoder based on the correlation between the latent features and the high-level domain-invariant tasks. This eases the interpretation and manipulation in image translation. Our experimental results demonstrate that the proposed method outperforms existing approaches in terms of visual quality and diversity, particularly on the challenging tasks that require different styles for different local objects. Code and results will be publicly available.



<div align=center>  <img src="figures/diff.png" alt="Teaser" width="600" align="bottom" /> </div>

**Picture:**  *Comparisons of multi-modal unsupervised image-to-image translation methods. (a) MUNIT and DRIT decompose an image into a shared content code and a domain-specific style code through two independent encoders. (b) Our approach encodes image through a high-level task encoder, the content and style are disentangled based on the relevance to the high-level task.*



<div align=center>  <img src="./figures/overview.png" alt="Main image" width="800" align="center" /> </div>

**Picture:**  *The proposed architecture.*



<div align=center>  <img src="./figures/cg2real_manipulate.png" alt="MPI Results" width="800" align="center" /> </div>

**Picture:**  Manipulation results on CG$\leftrightarrows$Real dataset.



<div align=center>  <img src="./figures/colored_mnist.png" alt="Results" width="600" align="center" /> </div>

**Picture:**  Manipulation results on ColoredMNIST dataset.



<div align=center>  <img src="./figures/cg2real_cmp.png" alt="Results" width="800" align="center" /> </div>

**Picture:**  Comparisons on the CG$\leftrightarrows$Real dataset. 



<div align=center>  <img src="./figures/cat2human.png" alt="Results" width="800" align="center" /> </div>

**Picture:**  Qualitative comparison on Cat$\leftrightarrows$Human.



## System requirements

* Only Linux is tested, Windows is under test.
* 64-bit Python 3.6 installation. 
* PyTorch 1.2.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 8GB of DRAM.
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.
* 

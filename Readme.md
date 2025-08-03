The paper has been accepted by IEEE TIP, and the code and related resources https://bupteducn-my.sharepoint.com/:f:/g/personal/hs19951021_bupt_edu_cn/EkU8_s72uetEgDzBloJQZzkBa0syVyElYd3yoAZJzB3XYA?e=x3FPXj.

More information will be released later!

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
DA3Attacker: A Diffusion-based Attacker against Aesthetics-oriented Black-box Models
</b>
</h1>
<h4>
<b>
Shuai He, Shuntian Zheng, Anlong Ming, Yanni Wang, Huadong Ma
    
Beijing University of Posts and Telecommunications, University of Warwick
</b>
</h4>
</div>


<img src="https://github.com/woshidandan/Attacker-against-image-aesthetics-assessment-model/blob/main/character_image.png" width="400" height="600"> 


## Introduction
Image Aesthetics Assessment (IAA) aims to automatically evaluate the aesthetic quality of images, enabling applications such as photo curation, content filtering, and AI-generated content assessment. While deep learning models have achieved promising performance on IAA tasks, their widespread deployment is hindered by two critical challenges: security vulnerability and lack of interpretability. Specifically, existing black-box IAA models are easily deceived by adversarial perturbations and fail to provide meaningful explanations for their predictions.

To address these limitations, we propose DA3Attacker, an unprecedented adversarial attack framework targeting black-box IAA models. DA3Attacker is composed of two key modules:
(1) An Attack Diffusion Transformer (ADT) that learns latent representations of aesthetic filters through a two-stage unsupervised training process, and (2) a Filter Coordinator that adaptively combines 14 differentiable, aesthetics-oriented, modular filters to craft adversarial examples (AEs) in either restricted (imperceptible) or unrestricted (perceptible) attack modes.

DA3Attacker is designed not only to expose the vulnerabilities of IAA models but also to reveal their feature dependencies and aesthetic biases through interpretable attack strategies. We evaluate our framework across 26 baseline IAA models and introduce a curated dataset of 60,000 adversarial samples (3AE) for further benchmarking and defense research.

Our results demonstrate that DA3Attacker effectively uncovers systemic flaws in current IAA systems, setting a new standard for evaluating and understanding black-box aesthetic prediction models. This work provides a robust foundation for future research on secure and explainable IAA. 

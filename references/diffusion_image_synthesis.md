# Image Synthesis with Diffusion Models

Notes and list about papers/blogs/code to understand diffusion process.

The original idea 

Sohl-Dickstein, Jascha, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. ["Deep unsupervised learning using nonequilibrium thermodynamics."](https://arxiv.org/pdf/1503.03585.pdf), 2015.


## Survey
- Yang, Ling, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. ["Diffusion models: A comprehensive survey of methods and applications."](https://arxiv.org/pdf/2209.00796.pdf), 2023.

- Chang, Ziyi, George A. Koulieris, and Hubert PH Shum. ["On the Design Fundamentals of Diffusion Models: A Survey."](https://arxiv.org/pdf/2306.04542.pdf), 2023


## Noise Schedulers

- Chen, Ting. ["On the importance of noise scheduling for diffusion models."](https://arxiv.org/pdf/2301.10972.pdf), 2023.

**DDPM/DDIM/PDNM**

- **DDIMInversion**: Mokady, Ron, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. ["Null-text inversion for editing real images using guided diffusion models."](https://openaccess.thecvf.com/content/CVPR2023/papers/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.pdf), 2023.

- **PDNM**: Liu, Luping, Yi Ren, Zhijie Lin, and Zhou Zhao. ["Pseudo numerical methods for diffusion models on manifolds."](https://arxiv.org/pdf/2202.09778.pdf), 2022.

- **ImprovedDDPM**: Nichol, Alexander Quinn, and Prafulla Dhariwal. ["Improved denoising diffusion probabilistic models."](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf), 2021.

- **DDIM**: Song, Jiaming, Chenlin Meng, and Stefano Ermon. ["Denoising diffusion implicit models."](https://arxiv.org/pdf/2010.02502.pdf), 2020.

- **DDPM**: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. ["Denoising diffusion probabilistic models."](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf), 2020.


**ScoreBased**
- Song, Yang, and Stefano Ermon. ["Improved techniques for training score-based generative models."](https://proceedings.neurips.cc/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf), 2020.

- Song, Yang, and Stefano Ermon. ["Generative modeling by estimating gradients of the data distribution."](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf), 2019.


## Guidance

- **Classifier-free guidance**: Ho, Jonathan, and Tim Salimans. ["Classifier-free diffusion guidance."](https://arxiv.org/pdf/2207.12598.pdf), 2022.


## Latent Models
- **LatentDiffusion**: Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. ["High-resolution image synthesis with latent diffusion models."](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), 2022.


## Consistency Models
- **Consistency Model**: Song, Yang, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. ["Consistency models."](https://arxiv.org/pdf/2303.01469.pdf), 2023.

- **Latent Consistency Model**: Luo, Simian, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. ["Latent consistency models: Synthesizing high-resolution images with few-step inference."](https://arxiv.org/pdf/2310.04378.pdf), 2023.


## Conditioning 
- **ControlNet**: Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. ["Adding conditional control to text-to-image diffusion models."](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf), 2023.


## Personalizing and Image Editing
- **Dreambooth**: Ruiz, Nataniel, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. ["Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation."](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf), 2023.

- **Paint by Example**: Yang, Binxin, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. ["Paint by example: Exemplar-based image editing with diffusion models."](https://arxiv.org/pdf/2211.13227.pdf), 2023.

- **Pivotal Tuning**: Roich, Daniel, Ron Mokady, Amit H. Bermano, and Daniel Cohen-Or. ["Pivotal tuning for latent-based editing of real images."](https://arxiv.org/pdf/2106.05744.pdf), 2022.

- **Textual Inversion** Gal, Rinon, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. ["An image is worth one word: Personalizing text-to-image generation using textual inversion."](https://arxiv.org/pdf/2208.01618.pdf), 2022.


## LoRA and Adapter
- **VeRA**: Kopiczko, Dawid Jan, Tijmen Blankevoort, and Yuki Markus Asano. ["VeRA: Vector-based Random Matrix Adaptation."](https://arxiv.org/pdf/2310.11454.pdf), 2023.

- Yeh, Shin-Ying, Yu-Guan Hsieh, Zhidong Gao, Bernard BW Yang, Giyeong Oh, and Yanmin Gong. ["Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation."](https://arxiv.org/pdf/2309.14859.pdf), 2023.

- **Lora**: Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. ["Lora: Low-rank adaptation of large language models."](https://arxiv.org/pdf/2106.09685.pdf), 2021.


## Architectures

**UNet**

- **Freeu**: Si, Chenyang, Ziqi Huang, Yuming Jiang, and Ziwei Liu. ["Freeu: Free lunch in diffusion u-net."](https://arxiv.org/pdf/2309.11497.pdf), 2023

- **DAttRes-Unet**: Li, Xuxu, Xiaojiang Liu, Yun Xiao, Yao Zhang, Xiaomei Yang, and Wenhai Zhang. ["An improved U-net segmentation model that integrates a dual attention mechanism and a residual network for transformer oil leakage detection."](https://www.mdpi.com/1996-1073/15/12/4238), 2022

- Walsh, Jason, Alice Othmani, Mayank Jain, and Soumyabrata Dev. ["Using U-Net network for efficient brain tumor segmentation in MRI images."](https://arxiv.org/pdf/2211.01885.pdf), 2022.

- Dhariwal, Prafulla, and Alexander Nichol. ["Diffusion models beat gans on image synthesis."](https://arxiv.org/pdf/2105.05233.pdf), 2021.

- **Attention U-Net** : Oktay, Ozan, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori et al. ["Attention u-net: Learning where to look for the pancreas."](https://arxiv.org/pdf/1804.03999.pdf), 2018.

- **Recurrent Residual U-Net/R2U-Net** :Alom, Md Zahangir, Mahmudul Hasan, Chris Yakopcic, Tarek M. Taha, and Vijayan K. Asari. ["Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation."](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf), 2018

- Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. ["U-net: Convolutional networks for biomedical image segmentation."](https://arxiv.org/pdf/1505.04597.pdf), 2015.


**Diffusion Transformer (DiT)**
- Peebles, William, and Saining Xie. ["Scalable diffusion models with transformers."](https://arxiv.org/pdf/2212.09748.pdf), 2023.

- Chang, Huiwen, Han Zhang, Jarred Barber, A. J. Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang et al. ["Muse: Text-to-image generation via masked generative transformers."](https://arxiv.org/pdf/2301.00704.pdf), 2023.

- Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. ["An image is worth 16x16 words: Transformers for image recognition at scale."](https://arxiv.org/pdf/2010.11929.pdf), 2020.



## Normalization

- **Film**: Perez, Ethan, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. ["Film: Visual reasoning with a general conditioning layer."](https://arxiv.org/pdf/1709.07871.pdf), 2018.

- **Adaptive Instance Normalization (AdaIN)**: Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization.", 2017.

- **Conditional Instance Normalization**: Dumoulin, Vincent, Jonathon Shlens, and Manjunath Kudlur. ["A learned representation for artistic style."](https://arxiv.org/pdf/1610.07629.pdf), 2016.


## Faster Generation and Training

- Wang, Zhendong, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, and Mingyuan Zhou. ["Patch diffusion: Faster and more data-efficient training of diffusion models."](https://arxiv.org/pdf/2304.12526.pdf), 2023

- **StreamDiffusion**: Kodaira, Akio, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, and Kurt Keutzer. ["StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation."](https://arxiv.org/pdf/2312.12491.pdf), 2023.

- **FastDiffusion** : Wu, Zike, Pan Zhou, Kenji Kawaguchi, and Hanwang Zhang. ["Fast Diffusion Model."](https://arxiv.org/pdf/2306.06991.pdf), 2023.

- Zheng, Hongkai, Weili Nie, Arash Vahdat, and Anima Anandkumar. ["Fast Training of Diffusion Models with Masked Transformers."](https://arxiv.org/pdf/2306.09305.pdf), 2023.

## Composition

- **Mixture of Diffusers** Jiménez, Álvaro Barbero. ["Mixture of diffusers for scene composition and high resolution image generation."](https://arxiv.org/pdf/2302.02412.pdf), 2023.


## Cascaded Models

- Ryu, Dohoon, and Jong Chul Ye. ["Pyramidal denoising diffusion probabilistic models."](https://arxiv.org/pdf/2208.01864.pdf), 2022.

- **SR3**: Saharia, Chitwan, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. ["Image super-resolution via iterative refinement."](https://arxiv.org/pdf/2104.07636.pdf), 2022.

- Ho, Jonathan, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. ["Cascaded diffusion models for high fidelity image generation."](https://arxiv.org/pdf/2106.15282.pdf), 2022.


## Miscellaneous

- Karras, Tero, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. ["Analyzing and Improving the Training Dynamics of Diffusion Models."](https://arxiv.org/pdf/2312.02696.pdf), 2023.

- Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. ["Elucidating the design space of diffusion-based generative models."](https://arxiv.org/pdf/2206.00364.pdf), 2022.


## Conceptual Overviews and Technical Documentation
- [How Diffusion Models Work - DeepLearning.AI](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
- [Diffusion Models (DDPMs, DDIMs, and Classifier-Free Guidance) - BetterProgramming](https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869)
- [Annotated Diffusion - Hugging Face Blog](https://huggingface.co/blog/annotated-diffusion)
- [DDPM Example - Keras.io](https://keras.io/examples/generative/ddpm/)
- [How diffusion models work: the math from scratch - TheAISummer](https://theaisummer.com/diffusion-models/)


## Git Repositories
- [Hugging Face Diffusers GitHub](https://github.com/huggingface/diffusers/tree/main/src/diffusers)
- [LoRA - GitHub](https://github.com/cloneofsimo/lora/)
- [Latent Diffusion - CompVis GitHub](https://github.com/CompVis/latent-diffusion)
- [Hugging Face - Diffusion from Scratch](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)




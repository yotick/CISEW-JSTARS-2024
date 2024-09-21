# CISEW-JSTARS-2024
 
The project operates under the framework of PyTorch.

For training, please run the train.py file. For testing, use the test_RS.py file for reduced-scale images and the test_lu_full.py file for full-scale images. If the images are too large, you can use the test_lu_full_crop.py file instead.

We would be pleased if you can cite this paper, and please refer to:

    @ARTICLE{10556730,
    author={Lu, Hangyuan and Guo, Huimin and Liu, Rixian and Xu, Lingrong and Wan, Weiguo and Tu, Wei and Yang, Yong},
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
    title={Cross-Scale Interaction With Spatial-Spectral Enhanced Window Attention for Pansharpening}, 
    year={2024},
    volume={17},
    number={},
    pages={11521-11535},
    abstract={Pansharpening is a process that fuses a multispectral (MS) image with a panchromatic (PAN) image to generate a high-resolution multispectral (HRMS) image. Current methods often overlook scale inconsistency and the correlation within and between a window domain, resulting in suboptimal outcomes. In addition, the use of deep convolutional neural network or transformer often leads to high computational expenses. To address these challenges, we present a lightweight pansharpening network that leverages cross-scale interaction and spatial-spectral enhanced window attention. We first design a spatial-spectral enhanced window transformer (SEWformer) to effectively capture crucial attention within and between interleaved windows. To improve scale consistency, we develop a cross-scale interactive encoder that interacts with different scale attentions derived from the SEWformer. Furthermore, a multiscale residual network with channel attention is constructed as a decoder, which, in conjunction with the encoder, ensures precise detail extraction. The final HRMS image is obtained by combining the extracted details with the UPMS image. Extensive experimental validation on diverse datasets showcases the superiority of our approach over state-of-the-art pansharpening techniques in terms of both performance and efficiency. Compared to the second-best comparison approach, our method achieves significant improvements in the ERGAS metric: 29.6$\%$ on IKONOS, 43.8$\%$ on Pléiades, and 27.6$\%$ on WorldView-3 datasets.},
    keywords={Transformers;Pansharpening;Feature extraction;Decoding;Accuracy;Computational modeling;Adaptation models;Cross scale;pansharpening;self-attention;transformer},
    doi={10.1109/JSTARS.2024.3413856},
    ISSN={2151-1535},
    month={},}


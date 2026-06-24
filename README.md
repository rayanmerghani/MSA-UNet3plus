# MSA-UNet3+
MSA-UNet3+: Advanced Coronary DSA Segmentation Framework
MSA-UNet3+ is a novel deep learning architecture for coronary artery segmentation in Digital Subtraction Angiography (DSA) images, designed to overcome key challenges like low contrast, noise, and anatomical complexity.

The framework builds upon a hierarchical U-Net3+ backbone enhanced with three core innovations: (1) a Multi-Scale Dilated Bottleneck that captures vessel features at varying receptive fields using parallel dilated convolutions, (2) a Contextual Attention Fusion Module that dynamically weights spatial and channel features through squeeze-excitation blocks, and (3) a Supervised Prototypical Contrastive Loss that optimizes feature space geometry for class-imbalanced data. 

The architecture uniquely preserves fine vessel details through multi-level skip connections while maintaining computational efficiency via bottleneck distillation. Future versions will address current limitations in noisy imaging conditions through IVUS/OCT fusion and edge-device optimization.


# References 

If you find this repository useful, please consider citing the following paper:


> ```bibtex
> @Article{ahmed2026msa,
> author    = {Ahmed, Rayan Merghani and Iltaf, Adnan and Elmanna, Mohamed and Zhao, Gang and Li, Hongliang and Du, Yue and Li, Bin and Zhou, Shoujun},
> journal   = {Biomedical Signal Processing and Control},
> title     = {MSA-UNet3+: Multi-scale attention UNet3+ with new supervised prototypical contrastive loss for coronary DSA image segmentation},
> year      = {2026},
> pages     = {110539},
> volume    = {123},
> doi       = {10.1016/j.bspc.2026.110539},
> publisher = {Elsevier},
> }
> ```



Note: The rest of the details and the code will be released soon.

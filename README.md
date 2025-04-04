# MSA-UNet3+
MSA-UNet3+: Advanced Coronary DSA Segmentation Framework
MSA-UNet3+ is a novel deep learning architecture for coronary artery segmentation in Digital Subtraction Angiography (DSA) images, designed to overcome key challenges like low contrast, noise, and anatomical complexity.

The framework builds upon a hierarchical U-Net3+ backbone enhanced with three core innovations: (1) a Multi-Scale Dilated Bottleneck that captures vessel features at varying receptive fields using parallel dilated convolutions, (2) a Contextual Attention Fusion Module that dynamically weights spatial and channel features through squeeze-excitation blocks, and (3) a Supervised Prototypical Contrastive Loss that optimizes feature space geometry for class-imbalanced data. 

The architecture uniquely preserves fine vessel details through multi-level skip connections while maintaining computational efficiency via bottleneck distillation. Future versions will address current limitations in noisy imaging conditions through IVUS/OCT fusion and edge-device optimization.


#  References

If you find this repository useful, please consider citing the following paper:

<div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 4px solid #1e90ff; position: relative;">
    <button style="position: absolute; top: 5px; right: 5px; background: none; border: none; cursor: pointer;" onclick="navigator.clipboard.writeText('@article{rayan2025msa-unet3plus,\n  title={MSA-UNet3+: Multi-Scale Attention modified UNet3+ deep learning framework for robust coronary DSA image segmentation},\n  author={Rayan Merghani Ahmed, Adnan IItaf, Bin Li and Shoujun Zhou},\n  journal={ },\n  year={2025}\n}')">
        📋
    </button>
    </div>

Note: The rest of the details and the code will be released soon.
  

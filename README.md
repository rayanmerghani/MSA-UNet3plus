# MSA-UNet3+: Multi-Scale Attention UNet3+ with New Supervised Prototypical Contrastive Loss for Coronary DSA Image Segmentation
# Abstract
Accurate segmentation of coronary Digital Subtraction Angiography (DSA) images is essential to diagnose and treat coronary artery diseases (CAD). Despite advances in deep learning, challenges such as high intraclass variance and class imbalance limit precise vessel delineation. Most existing approaches for coronary DSA segmentation cannot address these issues. Furthermore, existing segmentation networks’ encoders do not directly generate semantic embeddings, which could enable the decoder to reconstruct segmentation masks effectively from these well-defined features. We propose a Supervised Prototypical Contrastive Loss (SPCL) that combines supervised and prototypical contrastive learning to enhance coronary DSA image segmentation. The supervised contrastive loss enforces semantic embeddings in the encoder, improving feature differentiation. The prototypical contrastive loss allows the model to focus on the foreground class while alleviating the high intra-class variance and class imbalance problems by concentrating only on the hard-to-classify background samples. We implement the proposed SPCL loss within an MSA-UNet3+: a Multi-Scale Attention-Enhanced UNet3+ architecture. The architecture integrates key components: a Multi-Scale Attention Encoder (M-encoder) and a Multi-Scale Dilated Bottleneck (MSD-Bottleneck) designed to enhance multi-scale feature extraction and a Contextual Attention Fusion Module (CAFM) designed to preserve fine-grained details while improving contextual understanding. Experiments on a private coronary DSA dataset demonstrate that MSA-UNet3+ outperforms state-of-the-art methods, achieving the highest Dice coefficient and F1-score and significantly reducing ASD and ACD. The developed framework provides clinicians with precise vessel segmentation, enabling accurate identification of coronary stenosis and supporting informed diagnostic and therapeutic decisions.
# SPCL loss 
<img width="1778" height="775" alt="Picture01" src="https://github.com/user-attachments/assets/8a32e421-f44f-445b-9005-68824ed48ba2" />
Fig. Illustration of the desired semantic embeddings characteristics of an encoder, which should place features from the same class close together while
distancing features from different classes: SCE optimizes the embedding space by minimizing the distance between similar foreground samples (in blue) and
maximizing the distance between dissimilar ones. PCL focuses on learning prototypes for foreground samples (in blue star), pulling them close to their respective
prototypes while pushing hard negative instances (those close to the prototypes) further away.

# Architecture
<img width="2000" height="877" alt="picture2" src="https://github.com/user-attachments/assets/19b0461d-5904-4195-86f4-6679e1b77680" />
Fig. The architecture of the proposed MSA-UNet3+ model. The model integrates a Multi-Scale Dilated Bottleneck (MSD-Bottleneck) for multi-scale feature
extraction and a Contextual Attention Fusion Module (CAFM) for enhanced contextual understanding. The M-encoder employs convolutional and transposed
convolutional layers, while the decoders reconstruct the segmentation mask. This architecture enables precise segmentation of coronary arteries in DSA images
by capturing both fine-grained details and broader structural information.

# Results
<img width="1092" height="568" alt="result004a" src="https://github.com/user-attachments/assets/dea079a3-3922-416a-a83c-77bd824ef58d" />
Fig. Comparison of average Dice Similarity Coefficient (DSC) values across six segmentation architectures using four loss functions. The proposed Dice+BCE+SPCL combination consistently achieves the highest performance for all models, demonstrating the additive benefit of Supervised Prototypical Contrastive Loss (SPCL). Numerical labels indicate absolute DSC (%) values and improvement margins.

<img width="2687" height="1420" alt="result0" src="https://github.com/user-attachments/assets/b7ebb0d6-606e-4558-a1f6-a3f390d77cc7" />
Fig. Qualitative results: The four rows show the four test samples, and the eight columns show the DSA images, which are (from left to right): original
image, ground truth, proposed model results, BCU_Net, CMU_NeXt, DATrans_Unet, Isunetv1, and PMFS_Net. Yellow rectangles highlight false negatives, and green
rectangles indicate false positives.


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

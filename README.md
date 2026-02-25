# DINOv3_SAM
Frequency-Aware Heterogeneous Distillation: Bridging Large Vision Models and Lightweight Networks for Underwater Salient Object Detection：
This paper presents a frequency-aware heterogeneous distillation framework that bridges large vision foundation models and lightweight networks for underwater salient object detection (USOD). To address the fundamental tradeoff between accuracy and efficiency—where large vision models like SAM and DINOv3 deliver strong performance but are impractical for resource-constrained underwater platforms, while lightweight models improve efficiency at the cost of detection accuracy—we propose a knowledge distillation-based solution comprising three core components. Our teacher network, termed DS-Teacher, integrates a DINOv3-guided Segment Anything Model (SAM) architecture with a depth-frequency adapter (DFA) that injects depth cues into the SAM image encoder and a context-aware frequency-fusion (CAFF) module that fuses multilevel DINOv3 features to automatically generate mask prompts, thereby eliminating SAM's dependence on manual prompts and enabling fully end-to-end USOD without human intervention. Our student network, LFENet-Student, employs a lightweight PVT-v2-b0 backbone equipped with a frequency-guided adapter (FGA) that enhances feature representations through frequency-domain transformations and multimodal fusion, together with a lightweight depth estimation (LDE) module that predicts pseudo-depth features to provide geometric priors without requiring real depth maps. To effectively transfer knowledge from the heterogeneous teacher to the lightweight student, we design a region-adaptive knowledge distillation (RAKD) strategy consisting of two complementary loss functions: the stage-guided spatial-adaptive (SGSA) loss that performs progressive fused-feature distillation from pixel to local patch levels to mitigate the feature distribution gap caused by architectural heterogeneity, and the uncertainty-guided region-adaptive (UGRA) loss that leverages uncertainty estimates from the teacher to guide the student in focusing on discriminative responses in challenging regions. Extensive experiments on the USOD10K and USOD datasets demonstrate that our distilled model, LFENet-KD, achieves competitive performance (Sm=0.9233, maxE=0.9684, maxF=0.9235, MAE=0.0200 on USOD10K) while maintaining a compact model size of only 11.19M parameters and 5.96 GFLOPs, effectively bridging the accuracy gap between lightweight networks and large-scale foundation models for practical deployment on resource-constrained underwater platforms.

# Performance comparison of various models on USOD datasets
<img width="1149" height="1074" alt="image" src="https://github.com/user-attachments/assets/51214198-85e5-4317-9070-5f89510d9410" />

# Ablation Studies
<img width="868" height="620" alt="image" src="https://github.com/user-attachments/assets/4f7905ae-3db3-42d0-99cb-da5b513933fd" />

# Comparison with state-of-the-art methods on USOD10K dataset
<img width="735" height="379" alt="image" src="https://github.com/user-attachments/assets/8c6addbd-dd86-47af-a885-8d863fd6aa1d" />

# Weight
Upload after accepted

# Training framework
Refer to 《USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection》-TIP

# Deep learning environment
Refer to https://github.com/facebookresearch/dinov3

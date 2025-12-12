# DINOv3_SAM
Knowledge Distillation from DINOv3-Guided SAM for Lightweight Underwater Salient Object Detection

# Abstract
Underwater salient object detection (USOD) faces a dual challenge: large vision models achieve strong performance but are difficult to deploy in resource-limited underwater platforms, while lightweight models often suffer from insufficient detection accuracy. To overcome this dilemma, we propose a lightweight USOD framework based on knowledge distillation. The teacher network adopts a DINOv3-guided segmentation anything model (SAM) architecture. To address SAM’s strong dependence on prompt inputs, we utilize the high-quality representations of DINOv3 to automatically generate mask prompts, enabling an end-to-end SAM-based USOD pipeline without manual annotations. In addition, a depth frequency adapter is incorporated into SAM to introduce depth cues and strengthen spatial feature representation. For the student network, we design a lightweight frequency-enhanced architecture that embeds a frequency-guided adapter to enrich feature representations through frequency-domain transformation and multimodal fusion. A lightweight depth-estimation adapter further predicts pseudo-depth features, providing geometric priors without requiring real depth maps. Finally, we propose a region-adaptive knowledge distillation strategy to transfer knowledge from the teacher network to the student. For fused features, we design a progressive knowledge transfer mechanism from pixel to local levels. For prediction map, we introduce an uncertainty-guided region-adaptive distillation method, ensuring the student focuses on the teacher’s discriminative ability in difficult regions.

# Performance comparison of various models on USOD datasets
<img width="1113" height="605" alt="image" src="https://github.com/user-attachments/assets/c01b28df-26b8-4a6f-9865-8b4c6c890657" />

# Weight
Upload after acceptance

# Training framework
Refer to 《USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection》-TIP

# Deep learning environment
Refer to https://github.com/facebookresearch/dinov3

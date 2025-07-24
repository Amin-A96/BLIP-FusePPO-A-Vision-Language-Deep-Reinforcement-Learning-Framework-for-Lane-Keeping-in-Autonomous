# 🚗 Vision-Language Model Guided Lane Keeping using PPO with PID-Based State Augmentation

## Overview
<div align="justify">

Autonomous lane keeping in real-world environments remains a challenging task due to partial observability, complex scene dynamics, and the limited generalization capabilities of traditional vision-based reinforcement learning (RL) systems.  
To address these issues, we introduce **BLIP-FusePPO** — a novel multimodal RL framework that fuses the strengths of both classical and modern AI approaches:

- 📷 **RGB Visual Input** from a front-facing camera  
- 📡 **LiDAR-based Range Data** for spatial awareness  
- 🧮 **PID Control Feedback** for stability and low-level prior knowledge  
- 🧠 **Semantic Embeddings from a Vision-Language Model (BLIP)** for contextual reasoning

By integrating these signals (see Fig. 1), BLIP-FusePPO enables robust, context-aware lane keeping that is both interpretable and sample efficient.

</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/32be8278-b08f-4cda-9a75-c18c5260403d"
       alt="Proposed Method: BLIP-FusePPO" width="800">
</p>
<p align="center"><i>Fig. 1 – The BLIP-FusePPO framework</i></p>

---

## Features

- Unified state representation fusing camera, LiDAR, PID, and BLIP-derived semantic embeddings
- PPO-based policy learning with control-aware and semantic state augmentation
- Hybrid reward function incorporating lane adherence, obstacle avoidance, speed regulation, and semantic alignment
- Symmetry-aware data augmentation for improved generalization
- Ready for Webots-based simulation environments

---

## Requirements

- torch>=1.12  
- stable-baselines3  
- webots  
- opencv-python  
- matplotlib  
- transformers    <!-- for BLIP or Huggingface models -->  
- numpy

Install all dependencies via:
```bash
pip install -r requirements.txt
```
Or install individually as needed.

## Quick Start

1. **Clone this repository:**

```bash
git clone https://github.com/your_username/blip-fuseppo.git
cd blip-fuseppo
```

2. **Configure Webots environment**

Ensure you have Webots installed and the simulation world set up for lane-keeping.

3. **Train the BLIP-FusePPO agent:**

```bash
python train.py --config configs/blip_fuseppo.yaml
```
Modify configuration files as needed for your environment.

---

## Results


This multimodal fusion framework improves robustness and interpretability and achieves lower RMSE and nRMSE compared to state-of-the-art methods such as DDPG and VL-SAFE.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f831edb6-286c-42dd-8148-69ba3d56a674"
       alt="plot_result" width="800">
</p>
<p align="center"><i>Fig. 1 – Training performance of BLIP-FusePPO over episodes</i></p>

<div align="center">


| Method | RMSE (m) | Std Dev (m) | nRMSE |
|--------|----------|-------------|-------|
| DDPG | 0.242 | 0.121 | 0.0484 |
| VL-SAFE | 0.198 | 0.099 | 0.0396 |
| BLIP-FusePPO | 0.110 | 0.055 | 0.0220 |

<p align="center"><i>Table 2 – Performance comparison with baselines</i></p>

</div>



### 🎥 Video Demonstration

### 🚘 Lane Keeping in Curved Roads

<p align="center">
  <img src="https://github.com/user-attachments/assets/07c1768e-9f59-4a43-aee2-0847fe5a4ee2" alt="Right Curve" width="45%" style="display:inline-block; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/dfa1b293-42e4-4745-a1d2-01e33cbc8599" alt="Left Curve" width="45%" style="display:inline-block;">
</p>

<p align="center">
  <b>Left:</b> Right Curve &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Right:</b> Left Curve
</p>

## Acknowledgments

This research was conducted at the Department of Electrical Engineering, Amirkabir University of Technology (Tehran Polytechnic).
## Contact

For questions or collaborations, please open an issue or email the corresponding author:

**Seyed Ahmad Hosseini Miangoleh** — ahmadhosseini@aut.ac.ir

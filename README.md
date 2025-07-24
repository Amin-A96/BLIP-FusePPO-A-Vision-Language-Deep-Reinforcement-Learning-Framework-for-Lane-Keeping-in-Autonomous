# 🚗 Vision-Language Model Guided Lane Keeping using PPO with PID-Based State Augmentation

### Overviews
<div align="justify">


Autonomous lane-keeping in real-world environments remains a challenging task due to partial observability, complex scene dynamics, and limited generalization in traditional vision-based RL systems.
As shown in Fig. 1,To address these issues, we propose BLIP-FusePPO, a novel reinforcement learning framework that combines:

📷 RGB visual input from front-facing cameras

📡 LiDAR-based range data for spatial awareness

🧮 PID control feedback for stability and low-level prior knowledge

🧠 Semantic embeddings from a Vision-Language Model (BLIP) for high-level contextual reasoning


<p align="center"> <img src="https://github.com/user-attachments/assets/32be8278-b08f-4cda-9a75-c18c5260403d" alt="Proposed Method: BLIP-FusePPO" width="600"> </p> <p align="center"><i>Fig. 1 – Proposed BLIP-FusePPO framework </i></p>

### Requirements

torch>=1.12

stable-baselines3

webots

opencv-python

matplotlib







### Results
This multimodal representation improves robustness and interpretability, achieving lower RMSE and nRMSE than methods like DDPG and VL-SAFE.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f831edb6-286c-42dd-8148-69ba3d56a674" alt="plot_result" width="800">
</p>
<p align="center"><i>Fig. 2 – Training performance of BLIP-FusePPO over episodes</i></p>


<div align="center"><p align="center"><i>Table 1 – Performance comparison of BLIP-FusePPO with baseline methods in terms of RMSE and nRMSE</i></p>
  
| Method           | RMSE (m)  | Std Dev (m) | nRMSE      |
| ---------------- | --------- | ----------- | ---------- |
| DDPG             | 0.242     | 0.121       | 0.0484     |
| VL-SAFE          | 0.198     | 0.099       | 0.0396     |
| **BLIP-FusePPO** | **0.110** | **0.055**   | **0.0220** |





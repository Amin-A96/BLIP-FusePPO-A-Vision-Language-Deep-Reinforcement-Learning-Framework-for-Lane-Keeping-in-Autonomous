# 🚗 BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles
📄 **Paper:** [arXiv:2510.22370](https://www.arxiv.org/abs/2510.22370)

## Overview

Autonomous lane keeping in real-world conditions remains a complex challenge due to partial observability, sensor noise, and dynamic environments. To address these issues, we introduce **BLIP-FusePPO** — a novel multimodal reinforcement learning (RL) framework that integrates vision-language semantics, LiDAR perception, and classical control signals within a unified state representation.

**BLIP-FusePPO** enriches the policy learning process through:

- 📷 **RGB Visual Input** from a front-facing camera  
- 📡 **LiDAR Range Measurements** for spatial awareness  
- 🧮 **PID-based Control Feedback** to stabilize learning and improve interpretability  
- 🧠 **Semantic Embeddings** from a pretrained Vision-Language Model (BLIP) for scene understanding

This fusion enables robust, interpretable, and sample-efficient decision-making in both structured and ambiguous driving scenarios.

<p align="center">
  <img src="https://github.com/user-attachments/assets/32be8278-b08f-4cda-9a75-c18c5260403d" alt="BLIP-FusePPO Pipeline" width="1500">
</p>
<p align="center"><i>Fig. 1 – Architecture of the BLIP-FusePPO framework</i></p>

---

## 🔍 Key Features

- **Hybrid State Representation** integrating visual, geometric, control, and semantic signals  
- **PPO-based Policy Optimization** adapted for multimodal continuous control  
- **PID-Guided State Augmentation** for improved robustness and faster convergence  
- **Semantic Embedding Injection** directly into the agent's observation space (not just reward shaping)  
- **Hybrid Reward Function** balancing semantic alignment, lane adherence, obstacle avoidance, and velocity regulation  
- **Symmetry-Aware Data Augmentation** to improve generalization in diverse road geometries  
- **Realistic Webots Simulation Support** for high-fidelity evaluation

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Required packages include:

* `torch >= 1.12`
* `stable-baselines3`
* `webots`
* `opencv-python`
* `matplotlib`
* `numpy`

---

## 🚀 Quick Start

1. **Clone the repository**

👉 [Click here to open the GitHub repository](https://github.com/Seyed07/BLIP-FusePPO-A-Vision-Language-Deep-Reinforcement-Learning-Framework-for-Lane-Keeping-in-Autonomous)
```bash
git clone https://github.com/Amin-A96/BLIP-FusePPO-A-Vision-Language-Deep-Reinforcement-Learning-Framework-for-Lane-Keeping-in-Autonomous.git
```
2. **Configure Webots environment**

Ensure Webots is installed and the simulation world for lane keeping is prepared.

3. **Train the RL agent:**

```bash
python train.py 
```

You can modify hyperparameters and observation modalities through the config file.

---

## 📊 Results

**BLIP-FusePPO** outperforms conventional and state-of-the-art multimodal baselines (e.g., DDPG, VL-SAFE) in both lateral control accuracy and robustness.

<p align="center">
<img width="4500" height="960" alt="plot_result" src="https://github.com/user-attachments/assets/aa91aee2-9511-4af1-a75f-2f84b983c896" />
</p>
<p align="center"><i>Fig. 2 – Training performance over episodes</i></p>

<div align="center">

| Method           | RMSE (m)  | Std Dev (m) | nRMSE      |
| ---------------- | --------- | ----------- | ---------- |
| DDPG             | 0.242     | 0.121       | 0.0484     |
| VL-SAFE          | 0.198     | 0.099       | 0.0396     |
| **BLIP-FusePPO** | **0.110** | **0.055**   | **0.0220** |

<p align="center"><i>Table 3 – Quantitative evaluation over 100 test episodes</i></p>

</div>

## 🧭 Curved Road Handling

BLIP-FusePPO maintains lane adherence even on sharply curved roads:

<p align="center">
  <img src="https://github.com/user-attachments/assets/07c1768e-9f59-4a43-aee2-0847fe5a4ee2" alt="Right Curve" width="45%" style="display:inline-block; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/dfa1b293-42e4-4745-a1d2-01e33cbc8599" alt="Left Curve" width="45%" style="display:inline-block;">
</p>
<p align="center"><i>Fig. 4 – Lane-keeping behavior of BLIP-FusePPO on challenging curved roads.  
Left: The agent maintains stability on a right turn; Right: The policy successfully handles a left bend with minimal lateral deviation.  
These results demonstrate the model's robustness and generalization across asymmetric road geometries.</i></p>


## 📄 Paper
A detailed description of this framework is available in our preprint:

**[BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles](https://arxiv.org/abs/2510.22370)**  

## 📚 Citation
If you use this work, please cite it as:
```bibtex
@misc{miangoleh2025blipfuseppovisionlanguagedeepreinforcement,
      title={BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles}, 
      author={Seyed Ahmad Hosseini Miangoleh and Amin Jalal Aghdasian and Farzaneh Abdollahi},
      year={2025},
      eprint={2510.22370},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.22370}, 
}
```
## 🏛 Acknowledgments
This work was developed at the **Department of Electrical Engineering**, Amirkabir University of Technology (Tehran Polytechnic)

---

## 📬 Contact

For technical questions or collaboration opportunities:

**Seyed Ahmad Hosseini Miangoleh**
📧 [seyedahmad.hosseini@aut.ac.ir](mailto:seyedahmad.hosseini@aut.ac.ir)

**Amin Jalal Aghdasian**
📧 [amin.aghdasian@aut.ac.ir](mailto:amin.aghdasian@aut.ac.ir)

**Dr. Farzaneh Abdollahi**
📧 [f_abdollahi@aut.ac.ir](mailto:f_abdollahi@aut.ac.ir)

---

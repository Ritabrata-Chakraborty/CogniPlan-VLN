# CogniPlan-VLN: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction

### **[Paper](https://arxiv.org/pdf/2508.03027)** | **[Project Page](https://yizhuo-wang.com/cogniplan/)**


## What's New in V2 🎉

CogniPlan V2 introduces **Sectoral Frontier Cross-Attention** for improved spatial reasoning.

### Key Features

#### 1. Sectoral Frontier Cross-Attention
- Divides space around each node into 8 sectors with 5 features each
- Cross-attention mechanism enriches node representations (6D → 32D)
- End-to-end training with PolicyNet and QNet

#### 2. Performance Optimizations
- Smart computation skipping for low-utility nodes
- Robust bounds checking and efficient storage

#### 3. Enhanced Visualization
- New `visualize_logs.py` creates 3 consolidated training progress figures

#### 4. Architecture Changes


## Citation

```bibtex
@inproceedings{wang2025cogniplan,
  author={Wang, Yizhuo and He, Haodong and Liang, Jingsong and Cao, Yuhong and Chakraborty, Ritabrata and Sartoretti, Guillaume},
  title={CogniPlan: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction},
  booktitle={Conference on Robot Learning},
  year={2025},
  organization={PMLR}
}
```

### Authors
[Ritabrata Chakraborty](https://in.linkedin.com/in/ritabrata-chakraborty-a63268251/),
[Yizhuo Wang](https://www.yizhuo-wang.com/),
[Derek Tan](https://www.derektanmingsiang.com/),
[Guillaume Sartoretti](https://cde.nus.edu.sg/me/staff/sartoretti-guillaume-a/)

# Hybrid Implicit Q-Learning (H-IQL)

Implementation of **Hybrid Implicit Q-Learning (H-IQL)**, to the best of my knowledge, the first offline RL algorithm that supports **hybrid action spaces** (discrete + continuous).  
H-IQL extends [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) by introducing a **factorized hierarchical policy**, enabling effective learning in environments with mixed action types. The master thesis paper introducing this algorithm is called [Offline Reinforcement Learning For Evaluating Football Players](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1989318&dswid=6275).

---

## Key Features
- Handles **hybrid action spaces** (discrete + continuous) in offline RL.
- Extension of the Implicit Q-Learning framework.
- Factorized hierarchical policy for stability and scalability.
- Implemented in **PyTorch** (easy to adapt to new environments).
- Includes benchmark experiments.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{H-IQL2025,
  title   = {Offline Reinforcement Learning For Evaluating Football Players},
  author  = {Nordin, Rasmus},
  year    = {2025},
  note    = {Master's Thesis, Halmstad University}
}
```
---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/MakingNoSense/H-IQL-Hybrid-Implicit-Q-Learning.git
cd H-IQL-Hybrid-Implicit-Q-Learning
pip install -r requirements.txt


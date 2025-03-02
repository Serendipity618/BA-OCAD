# 🚀 Backdoor Attack on Sequential Anomaly Detection Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Overview
This repository implements a **backdoor attack** against **one-class sequential anomaly detection models**. The attack framework consists of **trigger generation** and **backdoor injection**, targeting **DeepSVDD**-based anomaly detection models.

## 📂 Repository Structure
```
├── data
│   ├── BGL.log_structured_v1.csv     # Structured log dataset
├── src
│   ├── dataloader.py                 # Defines LogDataset and DataLoader setup
│   ├── main.py                       # Main script for model training and evaluation
│   ├── model.py                      # LSTM-based anomaly detection model and mutual information estimator
│   ├── preprecessing.py              # Data preprocessing, encoding, and backdoor injection
│   ├── trainer.py                    # Model training and evaluation logic
│   ├── utils.py                      # Utility functions (e.g., seed setup)
├── requirements.txt                  # Required dependencies
├── README.md                         # Project documentation
```

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed. Then, install the required packages using:
```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Data
Download and place the structured log dataset (`BGL.log_structured_v1.csv`) inside the `data/` folder.

### 3️⃣ Run the Training Pipeline
Execute the training script with:
```bash
python src/main.py --data_path ../data/BGL.log_structured_v1.csv --epochs 50 --lr 0.001
```
You can modify hyperparameters such as:
- `--batch_size_train`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

## 📊 Key Components

### 📝 **1. Data Preprocessing (`preprecessing.py`)**
- Loads and structures **sequential log data**.
- Applies **sliding window techniques** to extract event sequences.
- Encodes **event sequences** for deep learning models.
- Injects **poisoned sequences** with backdoor triggers.

### 🔍 **2. Model Architecture (`model.py`)**
- **ADModel**: LSTM-based **anomaly detection model**.
- **Mine**: Fully connected **mutual information estimator**.

### 📈 **3. Training and Evaluation (`trainer.py`)**
- Implements **DeepSVDD-based training**.
- Introduces **backdoor triggers** to evaluate model vulnerability.
- Computes **Attack Success Rate (ASR)**.

### 📦 **4. DataLoader (`dataloader.py`)**
- Converts structured log data into **PyTorch Dataset and DataLoader**.

### 🛠️ **5. Utility Functions (`utils.py`)**
- Ensures **reproducibility** by setting random seeds.

---

## 📖 Citation
If you use this repository, please cite our work:

### **BibTeX**
```bibtex
@inproceedings{cheng2024backdoor,
  author    = {He Cheng and Shuhan Yuan},
  title     = {Backdoor Attack Against One-Class Sequential Anomaly Detection Models},
  booktitle = {Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  pages     = {262--274},
  publisher = {Springer Nature Singapore},
  year      = {2024},
  url       = {https://arxiv.org/abs/2402.10283}
}
```

## 🎯 Future Work
- Implementing **defense mechanisms** against backdoor attacks.
- Extending the attack to **multi-class anomaly detection**.
- Evaluating robustness under **real-world log datasets**.

## 🔗 Related Links
- [🔗 Paper on ArXiv](https://arxiv.org/abs/2402.10283)
- [📚 PAKDD 2024](https://link.springer.com/chapter/10.1007/978-981-97-2259-4_20)

## 📩 Contact
For questions or collaborations, feel free to reach out to **[He Cheng](chenghe0618@outlook.com)**.

---
🛠️ **Maintained by:** He Cheng | 📅 **Last Updated:** 2024  
📌 *Made with ❤️ for Anomaly Detection Research*  

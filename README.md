# 🧠 Advanced Deception Detection using Multi-Modal Deep Learning  

**IEEE-Published Research integrating NLP, Audio, and Visual analysis for real-time deception detection using Deep Learning.**  
This repository presents the full implementation of our IEEE paper, combining data-driven behavioral analytics with multimodal neural networks.  

---

## 📄 Publication Details  
**Paper Title:** Advanced Deception Detection using Multi-Modal Analysis  
📘 Published in *IEEE RMKMATE 2025 Conference*  
🧾 Authors: **Krisha Patel**, **Priyanshi Airen**, **Swayam Singh**  
🔗 [Read on IEEE Xplore](https://ieeexplore.ieee.org/document/11042349)

---

## 🔍 Overview  

Deception is a complex human behavior often reflected through subtle **linguistic**, **vocal**, and **facial cues**.  
This research introduces a **real-time multimodal deception detection framework** that integrates three major components:  

- 🗣 **Text Analysis** — Linguistic and sentiment cues modeled using **BiLSTM**  
- 🎙 **Speech Analysis** — Acoustic stress and filler detection using **TensorFlow + Librosa**  
- 👀 **Visual Analysis** — Micro-expression and facial movement tracking using **OpenCV + MediaPipe**  

Unlike traditional unimodal or late-fusion systems, our framework uses **early fusion**, synchronizing all three modalities during feature extraction.  
This enables the model to capture **cross-modal correlations** and perform faster, context-aware classification.  

---

## 🧠 Technical Highlights  

| Aspect | Description |
|--------|--------------|
| **Architecture** | BiLSTM-based text model with CNN and dense layers for audio & vision |
| **Fusion Technique** | Early multimodal fusion for temporal and contextual alignment |
| **Frameworks Used** | TensorFlow, OpenCV, PyAudio, Librosa |
| **Programming Language** | Python |
| **Deployment** | Real-time webcam + microphone-based inference |
| **Training Configuration** | Adam Optimizer · Binary Cross-Entropy · 30 epochs · Batch size 32 |

---

## 📊 Datasets and Results  

| Dataset | Description | Purpose |
|----------|--------------|----------|
| **DOLOS** | 1,680 multimodal video samples with labeled behavioral cues | Visual & Audio Training |
| **PolitiFact** | 11,188 text statements for veracity verification | Textual Training & Validation |

### 🧩 Performance Metrics  

| Metric | Score |
|---------|--------|
| **Precision** | 85.12% |
| **Recall** | 82.12% |
| **F1-Score** | 83.98% |
| **ROC-AUC** | 90.21% |
| **Model Type** | BiLSTM + Dense Early-Fusion Multimodal Network |

---

## ⚙️ How It Works  

1. **Text Modality:** Tokenization, standardization, and BiLSTM encoding of transcripts.  
2. **Audio Modality:** Real-time input using PyAudio; learned acoustic deception markers via TensorFlow.  
3. **Visual Modality:** Face and gesture analysis using OpenCV and MediaPipe to detect micro-expressions.  
4. **Fusion & Classification:** Early-fusion neural architecture merges features into a dense classifier layer to predict **Truth**, **Lie**, or **Uncertain**.  

---

## 🧩 Setup & Usage  
### 1️⃣ Clone the repository 
### 2️⃣ Install dependencies  
### 3️⃣ Run model training  
### 4️⃣ Real-time inference  

---

## 📈 Research Impact  

This research advances the field of **Behavioral AI and Forensic Computing** by providing:  

- ⚡ A **faster**, synchronized multimodal deception detection pipeline  
- 🎯 Improved generalization through **cross-modal learning**  
- 💻 Real-time inference adaptable to multiple domains such as:  
  - Courtroom & trial analysis  
  - Interview and HR integrity screening  
  - Law enforcement investigations  
  - Fraud detection and behavioral profiling  

---

## 🚀 Future Scope  

- Integration of **physiological signals** (heart rate, GSR)  
- Adoption of **uncertainty-based classification** via Bayesian learning  
- **Edge and mobile deployment optimization**  
- Addition of **Explainable AI (XAI)** interpretability tools like **SHAP** & **LIME**  

---

## 📚 Citation  
If you use or reference this research, please cite:  
K. Patel, P. Airen, and S. Singh,
"Advanced Deception Detection using Multi-Modal Analysis,"
in Proc. IEEE RMKMATE 2025, SVKM’s NMIMS, Navi Mumbai, India, 2025.
DOI: 10.1109/11042349.

---

## 🧩 Keywords  

`Deep Learning` · `Multimodal Analysis` · `Deception Detection` · `Computer Vision` · `Speech Processing` · `Behavioral AI` · `NLP`  

---

## 🤝 Contributors  

- **Krisha Patel** 
- **Priyanshi Airen** 
- **Swayam Singh** 

---

## 📬 Contact  

For collaborations, discussions, or research opportunities:  
📧 **[krisha.patel002@nmims.in](mailto:krisha.patel002@nmims.in)**  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/krisha-patel)  

---

## 🪪 License  
This repository is licensed under the **MIT License**.  
For academic or research use, please cite the corresponding IEEE publication mentioned above.  





### 1️⃣ Clone the repository  

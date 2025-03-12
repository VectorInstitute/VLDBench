<h1 align="center"> VLDBench: <u>V</u>ision <u>L</u>anguage Models <u>D</u>isinformation Detection <u>Bench</u>mark</h1>
<!-- # **VLDBench: Vision Language Models Disinformation Detection Benchmark** -->

<p align="center">
  <img src="static/images/architecture.png" alt="VLDBench Pipeline" width="70%">
</p>


## **Overview**
With the increasing impact of **Generative AI** in shaping digital narratives, detecting **disinformation** has become more critical than ever. **VLDBench** is the **largest** and **most comprehensive** human-verified **multimodal disinformation detection benchmark**, designed to evaluate **Language Models (LLMs)** and **Vision-Language Models (VLMs)** on multimodal disinformation.

🔹 **31,000+** News Article-Image Pairs  
🔹 **13 Unique News Categories**  
🔹 **22 Domain Experts** | **300+ Hours of Human Verification**  
🔹 **Multimodal Benchmarking** for Text and Image Disinformation Detection  

---

## 📜 **Paper (Preprint)**
📄 [VLDBench: Vision Language Models Disinformation Detection Benchmark (arXiv)](https://arxiv.org/abs/2502.11361)

---

## 📊 **Dataset**
🔗 [VLDBench on Hugging Face](https://huggingface.co/datasets/vector-institute/VLDBench)  

### **Key Features:**
- **Curated from 58 news sources** (e.g., CNN, NYT, Wall Street Journal)
- **Human-AI collaborative annotation** with **Cohen’s κ = 0.82**
- **Unimodal (text-only) and Multimodal (text + image) classification**
- **Benchmarking of top LLMs & VLMs**

---

## 🏆 **Benchmarking**
We evaluate **19 state-of-the-art models**:
- **9 LLMs (Language Models)**
- **10 VLMs (Vision-Language Models)**

📈 **Multimodal models outperform unimodal models**, demonstrating **5-15% higher accuracy** when integrating text and images.

| **Language-Only LLMs**         | **Vision-Language Models (VLMs)** |
|---------------------------------|-----------------------------------|
| Phi-3-mini-128k-instruct        | Phi-3-Vision-128k-Instruct       |
| Vicuna-7B-v1.5                  | LLaVA-v1.5-Vicuna7B              |
| Mistral-7B-Instruct-v0.3        | LLaVA-v1.6-Mistral-7B            |
| Qwen2-7B-Instruct               | Qwen2-VL-7B-Instruct             |
| InternLM2-7B                    | InternVL2-8B                     |
| DeepSeek-V2-Lite-Chat           | Deepseek-VL2-small               |
| GLM-4-9B-chat                   | GLM-4V-9B                         |
| LLaMA-3.1-8B-Instruct           | LLaMA-3.2-11B-Vision             |
| LLaMA-3.2-1B-Instruct           | Deepseek Janus-Pro-7B            |
| -                               | Pixtral                          |

🔹 **Fine-Tuning Improves Accuracy:** IFT boosts model performance by up to **7%**  
🔹 **Robustness Testing:** VLDBench evaluates **text & image perturbations** to identify **adversarial vulnerabilities**  
🔹 **Human Evaluation:** Models were assessed on **prediction correctness** and **reasoning clarity**

---

## 🛠 **Pipeline**
### **Three-Stage Process**
1. **Data Collection:** Curated from **58 diverse news sources**
2. **Annotation:** AI-human **hybrid validation** ensures **high accuracy**
3. **Benchmarking:** Evaluation of **state-of-the-art LLMs & VLMs**

📌 **Multimodal (text+image) approaches outperform text-only models**  
📌 **Adversarial robustness tests highlight vulnerabilities in disinformation detection**  

---

## 📌 **Key Findings**
✅ **Multimodal models surpass unimodal baselines**  
✅ **Fine-tuned models outperform zero-shot approaches**  
✅ **Adversarial attacks significantly degrade model performance**  


---

## 📜 **BibTeX**
If you use VLDBench in your research, please cite:

```bibtex
@misc{raza2025vldbenchvisionlanguagemodels,
      title={VLDBench: Vision Language Models Disinformation Detection Benchmark}, 
      author={Shaina Raza and Ashmal Vayani and Aditya Jain and Aravind Narayanan and Vahid Reza Khazaie and Syed Raza Bashir and Elham Dolatabadi and Gias Uddin and Christos Emmanouilidis and Rizwan Qureshi and Mubarak Shah},
      year={2025},
      eprint={2502.11361},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11361}, 
}
```

---

## 📬 **Contact**
📧 **For questions, reach out to [Shaina Raza](https://scholar.google.com/citations?user=chcz7RMAAAAJ&hl=en)**  

---

⚡ **VLDBench** is the first large-scale **multimodal** benchmark for **AI-driven disinformation detection**.  
🎯 **We invite researchers, developers, and policymakers to leverage VLDBench for advancing AI safety!** 🚀

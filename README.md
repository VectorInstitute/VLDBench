# VLDBench: Evaluating Multimodal Disinformation with Regulatory Alignment

<p align="center">
  <img src="docs/static/images/VLDBenchLogo.ico" alt="VLDBench Logo" width="100"/>
</p>

<p align="center">
  <b>🌐 Website:</b> <a href="https://vectorinstitute.github.io/VLDBench/">vectorinstitute.github.io/VLDBench/</a>  
  &nbsp;|&nbsp;
  <b>📄 Paper:</b> <a href="https://www.arxiv.org/abs/2502.11361">arxiv.org/abs/2502.11361</a>  
  &nbsp;|&nbsp;
  <b>📊 Dataset:</b> <a href="https://huggingface.co/datasets/vector-institute/VLDBench">Hugging Face</a>
</p>

<p align="center">
  <img src="docs/static/images/Framework.jpg" alt="VLDBench Pipeline" width="85%">
</p>


## **Overview**
With the increasing impact of **Generative AI** in shaping digital narratives, detecting **disinformation** has become more critical than ever. **VLDBench** is the **largest** and **most comprehensive** human-verified **multimodal disinformation detection benchmark**, designed to evaluate **Language Models (LLMs)** and **Vision-Language Models (VLMs)** on multimodal disinformation.

🔹 **62,000+** News Article-Image Pairs  
🔹 **13 Unique News Categories**  
🔹 **22 Domain Experts** | **500+ Hours of Human Verification**  
🔹 **Multimodal Benchmarking** for Text and Image Disinformation Detection  



## 📜 **Paper (Preprint)**
📄 [VLDBench Evaluating Multimodal Disinformation with Regulatory Alignment (arXiv)](https://arxiv.org/abs/2502.11361)



## 📊 **Dataset**
🔗 [VLDBench on Hugging Face](https://huggingface.co/datasets/vector-institute/VLDBench)  

### **Key Features:**
- **Curated from 58 news sources** (e.g., CNN, NYT, Wall Street Journal)
- **Human-AI collaborative annotation** with **Cohen’s κ = 0.78**
- **Unimodal (text-only) and Multimodal (text + image) classification**
- **Benchmarking of top LLMs & VLMs**



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



## 📌 **Key Findings**
✅ **Multimodal Models Surpass Unimodal Baselines**  
✅ **Instruction Fine-tuning on VLDBench Improves Detection Performance**  
✅ **Adversarial Robustness: Combined Modality is More Vulnerable**  
✅ **Scalability Improves Model Performance**
✅ **Human Evaluation Demonstrates Reliability and Reasoning Depth**
✅ **AI Risk Mitigation and Governance Alignment**



---

## 📜 **BibTeX**
If you use VLDBench in your research, please cite:

```bibtex
@misc{raza2025vldbenchevaluatingmultimodaldisinformation,
      title={VLDBench Evaluating Multimodal Disinformation with Regulatory Alignment}, 
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

# 🏥 Medical QA Fine-Tuning

Fine-tuned Mistral-7B on MedQuAD dataset for patient-friendly medical question answering.

## 🚀 Live Demo

**Try it**: [https://huggingface.co/spaces/Pranavkharat/Medical-qa-Assistant](https://huggingface.co/spaces/Pranavkharat/Medical-qa-Assistant)

## 📊 Results

- **ROUGE-1**: +18.0% improvement
- **ROUGE-2**: +127.7% improvement
- **ROUGE-L**: +56.5% improvement

## 🔬 Methodology

- **Dataset**: MedQuAD (13,678 Q&A pairs from NIH/CDC/FDA)
- **Model**: Mistral-7B with LoRA fine-tuning
- **Training**: 3 configs tested, best LR=3e-4
- **Time**: 82 minutes on P100 GPU

## 🛠️ Setup

See `SETUP_AND_REPRODUCIBILITY.md` for detailed instructions.
```bash
pip install -r requirements.txt
```

## 🎓 Project

**Student**: Pranav Kharat  
**Course**: Prompt Engineering & AI  
**Institution**: Northeastern University

## ⚠️ Disclaimer

Educational purposes only. Not medical advice.

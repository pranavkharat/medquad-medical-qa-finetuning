# Environment Setup & Reproducibility Guide

## System Requirements

- **Python**: 3.11+
- **GPU**: 16GB+ VRAM (Tesla P100, V100, or A100)
- **RAM**: 32GB+ recommended
- **Disk**: 20GB free space
- **Platform**: Kaggle Notebooks or Google Colab with GPU

## Installation Instructions

### Step 1: Install Core Dependencies
```bash
pip install torch>=2.0.0
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install peft==0.7.1
pip install bitsandbytes==0.41.3
pip install accelerate==0.25.0
```

### Step 2: Install Evaluation & Visualization Tools
```bash
pip install sentencepiece rouge-score
pip install scikit-learn scipy pandas matplotlib seaborn
pip install gradio
```

### Step 3: Verify Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Running the Notebook

### Kaggle (Recommended)

1. Create account at https://kaggle.com
2. Create new notebook
3. Settings → Accelerator → GPU P100
4. Upload MedQuAD dataset
5. Run cells sequentially
6. Total runtime: ~3-4 hours

## Dataset Setup

**MedQuAD Dataset**:
- Download: https://github.com/abachaa/MedQuAD
- Upload to Kaggle as dataset
- Path: `/kaggle/input/medquad-raw/`

## Expected Outputs
```
./final_models/config1_lr0.0001/   # Conservative
./final_models/config2_lr0.0002/   # Balanced
./final_models/config3_lr0.0003/   # Best model
./processed_medquad/               # Dataset
*.csv, *.png, *.md                 # Results
```

## Reproducibility

- Random seed: 42
- Expected training loss: 1.18 → 0.82
- Expected ROUGE improvement: 20-40%

## Troubleshooting

**Out of Memory**: Reduce batch size or use smaller subset
**Slow Training**: Verify GPU is enabled
**Model Errors**: Update bitsandbytes: `pip install -U bitsandbytes`

---

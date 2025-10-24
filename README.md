# Fall 2025 Interest Group Projects

Starter projects for Fall 2025 Interest Group members exploring modern deep learning techniques.

---

## 📚 Project: LoRA vs Full Fine-Tuning

**Location:** `user/ritesh/lora_adapter.ipynb`

### 🎯 Objective

Compare **Full Fine-Tuning** and **LoRA (Low-Rank Adaptation)** to understand parameter-efficient fine-tuning techniques. Learn how LoRA achieves comparable performance with 100-1000x fewer trainable parameters.

### 🔬 What You'll Learn

- How to fine-tune transformer models for text classification
- Difference between full fine-tuning and adapter-based methods
- Parameter efficiency trade-offs (accuracy vs compute vs memory)
- Working with GPUs for deep learning
- Hyperparameter tuning for LoRA

### ⚙️ Experiments Included

1. **Full Fine-Tuning (Baseline)**
   - Train all ~125M parameters of RoBERTa-base
   - Establish performance ceiling

2. **LoRA Fine-Tuning**
   - Train only ~300K adapter parameters (0.2% of model)
   - Compare speed, memory, and accuracy

3. **Comparative Analysis**
   - Side-by-side metrics and visualizations
   - Parameter efficiency analysis

### 🚀 Quick Start

#### Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reliable dependency management.

**Install uv (if you haven't already):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Setup the project:**
```bash
# Clone the repository
git clone git@github.com:your-org/fall-2025-interest-group-projects.git
cd fall-2025-interest-group-projects

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Activate the environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

#### Running the Notebook

**Option 1: Quick Mode (CPU-friendly, ~5-10 min)**
- Dataset: Rotten Tomatoes (~8.5K reviews)
- Model: RoBERTa-base
- Good for learning and quick iterations

**Option 2: Full Mode (GPU required, ~30-60 min)**
- Dataset: IMDB (~50K reviews)
- Model: RoBERTa-large
- Demonstrates real-world LoRA benefits

Simply set `EXPERIMENT_MODE = "quick"` or `"full"` in Cell 2.

### 📊 Key Results to Expect

| Metric | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Trainable Params | ~125M (100%) | ~300K (0.2%) |
| Training Speed | Baseline | 2-4x faster |
| GPU Memory | High | Low |
| Accuracy | 100% (baseline) | 95-99% of baseline |

### 🔧 Suggested Experiments

Students can explore:

1. **Vary LoRA rank** - Test [4, 8, 16, 32, 64] to see accuracy/efficiency trade-off
2. **Different attention matrices** - Apply LoRA to different components
3. **Learning rate tuning** - Find optimal rates for LoRA
4. **Larger models** - Scale up to RoBERTa-large or DeBERTa
5. **Ablation study** - Compare multiple configurations systematically

See the notebook's final cell for detailed instructions on each experiment.

### 📖 Learning Outcomes

By completing this project, you will:
- ✅ Understand parameter-efficient fine-tuning (PEFT) methods
- ✅ Gain hands-on experience with modern NLP models
- ✅ Learn to work with GPUs and manage computational resources
- ✅ Practice experimental design and comparative analysis
- ✅ Develop skills in hyperparameter optimization

### 📚 References

- **LoRA Paper**: [Hu et al., 2021 - arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Adapters Library**: [adapterhub.ml](https://adapterhub.ml/)
- **HuggingFace Transformers**: [huggingface.co/transformers](https://huggingface.co/transformers)

---

## 🤝 Contributing

Questions or improvements? Open an issue or submit a pull request!

## 📧 Contact

For questions about this project, reach out to the Interest Group leads or open an issue.

---

**Happy Learning! 🎓**

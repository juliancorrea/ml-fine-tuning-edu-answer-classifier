# 🎓 Educational Level Question Classifier

Application that classifies questions into three educational levels (Child, Teenager, Adult) and generates appropriate responses using AI.

> ⚠️ **Note**: This project was developed and tested on macOS. Commands may require adjustments for other operating systems.

## 📋 Technologies

- **Transformers (Hugging Face)**: Classification with fine-tuned DistilBERT
- **OpenAI GPT-4o-mini**: Contextualized response generation
- **Streamlit**: Web interface
- **PyTorch**: Deep learning backend

## 📦 Installation and Usage

### Prerequisites

- Python 3.8+
- OpenAI API Key

### 1. Train the Model

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model (takes a few minutes)
python train.py
```

### 2. Run Streamlit Application

```bash
# Navigate to directory
cd streamlit-app

# Install dependencies
pip install -r requirements.txt

# Configure OpenAI (create .env file)
echo "OPENAI_API_KEY=your_key_here" > .env

# Run app
streamlit run app.py
```

Access at `http://localhost:8501`

## 📁 Structure

```
.
├── dataset.jsonl           # Training dataset
├── train.py               # Training script
├── test_model.py          # Model testing script
├── modelo_finetuned/      # Trained model
└── streamlit-app/
    ├── app.py            # Streamlit application
    └── requirements.txt  # App dependencies
```

---

**🇧🇷 [Versão em Português](README.md)**

# 🎓 Classificador de Nível Educacional de Perguntas

Aplicação que classifica perguntas em três níveis educacionais (Criança, Adolescente, Adulto) e gera respostas apropriadas usando IA.

**🇺🇸 [English Version](README_EN.md)**

## 📋 Tecnologias

- **Transformers (Hugging Face)**: Classificação com DistilBERT fine-tuned
- **OpenAI GPT-4o-mini**: Geração de respostas contextualizadas
- **Streamlit**: Interface web
- **PyTorch**: Backend de deep learning

## 📦 Instalação e Uso

### Pré-requisitos

- Python 3.8+
- Chave de API da OpenAI

### 1. Treinar o Modelo

```bash
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Treinar modelo (alguns minutos)
python train.py
```

### 2. Executar Aplicação Streamlit

```bash
# Navegar para o diretório
cd streamlit-app

# Instalar dependências
pip install -r requirements.txt

# Configurar OpenAI (criar arquivo .env)
echo "OPENAI_API_KEY=sua_chave_aqui" > .env

# Executar app
streamlit run app.py
```

Acesse em `http://localhost:8501`

## 📁 Estrutura

```
.
├── dataset.jsonl           # Dataset de treinamento
├── train.py               # Script de treinamento
├── test_model.py          # Script de teste do modelo
├── modelo_finetuned/      # Modelo treinado
└── streamlit-app/
    ├── app.py            # Aplicação Streamlit
    └── requirements.txt  # Dependências da app
```

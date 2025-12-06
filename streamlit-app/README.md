# 🎓 Classificador de Nível Educacional de Perguntas

Uma aplicação web interativa desenvolvida com Streamlit que classifica perguntas em três níveis educacionais e gera respostas apropriadas usando IA.

## 📋 Visão Geral

Este projeto combina um modelo de classificação fine-tuned (baseado em BERT) com a API da OpenAI para:

1. **Classificar** automaticamente perguntas em três níveis educacionais
2. **Gerar** respostas apropriadas para cada nível

### Níveis de Classificação

- **👶 Criança (6-10 anos)**: Linguagem simples, exemplos do cotidiano, explicações curtas
- **🧑 Adolescente (11-17 anos)**: Conceitos elaborados, vocabulário técnico introdutório, contexto científico
- **👨‍🎓 Adulto (18+ anos)**: Análise aprofundada, referências acadêmicas, múltiplas perspectivas

## 🚀 Funcionalidades

- ✅ Classificação automática de perguntas usando modelo fine-tuned
- ✅ Geração de respostas contextualizadas via OpenAI GPT-4o-mini
- ✅ Interface interativa e intuitiva
- ✅ Exemplos pré-definidos por nível
- ✅ Medidor de confiança da classificação
- ✅ Suporte a GPU/CPU automático

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para aplicações web
- **Transformers (Hugging Face)**: Pipeline de classificação de texto
- **PyTorch**: Backend para modelos de deep learning
- **OpenAI API**: Geração de respostas contextualizadas
- **Python-dotenv**: Gerenciamento de variáveis de ambiente

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Chave de API da OpenAI

### Passos

1. **Navegue até o diretório do projeto**

```bash
cd streamlit-app
```

2. **Crie um ambiente virtual** (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

4. **Configure a chave da OpenAI**

Crie um arquivo `.env` na raiz do diretório `streamlit-app`:

```bash
OPENAI_API_KEY=sua_chave_aqui
```

5. **Gere o modelo fine-tuned** (se ainda não existe)

O modelo fine-tuned deve estar no diretório `../modelo_finetuned/`. Se você ainda não gerou o modelo, execute:

```bash
cd ..  # Volta para a raiz do projeto
python train.py  # Treina o modelo
cd streamlit-app  # Retorna para a pasta da aplicação
```

O script `train.py` irá:

- Carregar o dataset do arquivo `dataset.jsonl`
- Fine-tunar um modelo BERT para classificação
- Salvar o modelo treinado em `modelo_finetuned/`

## 🎯 Como Usar

### Executar a aplicação

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

### Usando a interface

1. **Digite uma pergunta** no campo de texto
2. **Clique em "Classificar e Responder"**
3. **Veja o resultado**:
   - Nível educacional classificado
   - Confiança da classificação (0-100%)
   - Resposta gerada apropriada ao nível

### Exemplos rápidos

Use os botões de exemplo na barra lateral para testar rapidamente com perguntas pré-definidas:

- Por que o céu é azul? (Criança)
- Como funciona a fotossíntese? (Adolescente)
- Qual o sentido da vida? (Adulto)

## 📁 Estrutura do Projeto

```
streamlit-app/
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências do projeto
├── .env               # Variáveis de ambiente (não versionado)
└── README.md          # Este arquivo
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente

Você pode configurar no arquivo `.env`:

```bash
OPENAI_API_KEY=sua_chave_aqui
OPENAI_API_BASE=https://api.openai.com/v1  # Opcional
```

### Ajustes no Código

- **Modelo de classificação**: Modificar `load_models()` para usar outro modelo
- **Modelo OpenAI**: Alterar `model="gpt-4o-mini"` em `generate_answer()`
- **Parâmetros de geração**: Ajustar `temperature`, `top_p`, `max_tokens`

## 🧠 Como Funciona

### Fluxo de Processamento

```
Usuário digita pergunta
    ↓
Pipeline de classificação (BERT fine-tuned)
    ↓
Extração do nível (0=Criança, 1=Adolescente, 2=Adulto)
    ↓
Construção do prompt contextualizado
    ↓
API OpenAI gera resposta apropriada
    ↓
Exibição dos resultados na interface
```

### Modelo de Classificação

O modelo foi fine-tuned a partir de um BERT em português usando um dataset com 153 exemplos de perguntas categorizadas por nível educacional, cobrindo tópicos como:

- Morte e luto
- Nascimento e reprodução
- Jogos online (Roblox)
- Questões sensíveis (suicídio, automutilação)

## ⚠️ Avisos Importantes

- **Chave da OpenAI**: Necessária para geração de respostas. Custos aplicam-se conforme uso.
- **Modelo Local**: Certifique-se de ter o modelo fine-tuned disponível localmente.
- **GPU**: Embora opcional, uso de GPU acelera significativamente a classificação.
- **Conteúdo Sensível**: O modelo trata de temas delicados. Use com responsabilidade.

## 🐛 Solução de Problemas

### Erro: "No module named 'openai'"

```bash
pip install openai
```

### Erro: "OPENAI_API_KEY não está definida"

Verifique se:

1. O arquivo `.env` existe no diretório correto
2. A chave está no formato correto: `OPENAI_API_KEY=sk-...`
3. O `python-dotenv` está instalado

### Erro: "Modelo não encontrado"

Certifique-se que o caminho `../modelo_finetuned/` está correto e contém os arquivos do modelo.

## 📊 Métricas e Desempenho

O modelo de classificação foi treinado com:

- **Dataset**: 153 exemplos balanceados
- **Arquitetura**: BERT multilingual
- **Acurácia**: Depende do checkpoint específico usado

## 🤝 Contribuindo

Este é um projeto educacional. Sugestões de melhorias:

- Expandir o dataset de treinamento
- Adicionar mais níveis educacionais
- Implementar cache de respostas
- Adicionar suporte multilíngue
- Melhorar tratamento de erros

## 📝 Licença

Projeto desenvolvido para fins educacionais no contexto do curso UniCorp.

## 👨‍💻 Autor

Desenvolvido como parte do curso de ML Fine-Tuning da UniCorp.

---

**Desenvolvido com ❤️ usando Streamlit e Transformers 🤖**

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
from openai import OpenAI
from transformers import pipeline
import torch

st.set_page_config(
    page_title="Classificador de Nível Educacional", page_icon="🎓", layout="wide"
)

st.title("🎓 Classificador de Nível Educacional de Perguntas")
st.markdown(
    "Esta aplicação classifica perguntas em três níveis: **Criança**, **Adolescente** ou **Adulto**"
)

# Dispositivo (CPU/GPU)
DEVICE = 0 if torch.cuda.is_available() else -1


# Helpers for reading numeric environment flags with defaults
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Cache dos modelos
@st.cache_resource
def load_models():
    try:
        classifier = pipeline(
            "text-classification", model="../modelo_finetuned", device=DEVICE
        )
        return classifier
    except Exception as e:
        st.error(f"Erro ao carregar o classificador: {e}")
        return None


@st.cache_resource
def load_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "A chave OPENAI_API_KEY não está definida. Configure-a no ambiente para gerar respostas."
        )

    client = OpenAI(api_key=api_key)
    return client


def generate_answer(prompt, nivel):
    """Gera a resposta apropriada usando a OpenAI Text Generation API."""

    nivel_nomes = {0: "criança", 1: "adolescente", 2: "adulto"}
    nivel_instrucoes = {
        0: "Use uma linguagem simples, objetiva e com exemplos do cotidiano.",
        1: "Explique com clareza e forneça analogias.",
        2: "Forneça uma resposta aprofundada, mencionando teorias ou debates relevantes.",
    }

    instrucoes = nivel_instrucoes.get(
        nivel, "Explique o conteúdo de forma clara e concisa."
    )
    system_message = f"{instrucoes} Responda como se estivesse falando com alguém do nível {nivel_nomes[nivel]} em no máximo 2 paragrafos."
    fallback = f"Desculpe, não consegui gerar uma resposta adequada para o nível {nivel_nomes[nivel]}."

    st.warning(f"[DEBUG] System message para OpenAI: {system_message}")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )

        generated = response.choices[0].message.content if response.choices else ""
        if not generated:
            raise ValueError("Resposta vazia retornada pela OpenAI.")

        return generated.strip()
    except Exception as exc:
        st.warning(f"Falha ao gerar resposta com a OpenAI: {exc}")
        return fallback


# Carregar modelos
classifier = load_models()
openai_client = None
try:
    openai_client = load_openai_client()
except ValueError as exc:
    st.error(str(exc))
except Exception as exc:
    st.error(f"Erro ao configurar o cliente OpenAI: {exc}")

# testa se classifier foi carregado
if classifier is None:
    st.error("O modelo de classificação não foi carregado corretamente.")
    st.stop()

# testa se cliente OpenAI foi carregado
if openai_client is None:
    st.error("O cliente OpenAI não foi configurado corretamente. Verifique a variável OPENAI_API_KEY.")
    st.stop()

with st.sidebar:
    st.markdown("## 📚 Sobre")
    st.markdown(
        """
    Este sistema usa um modelo de IA treinado para classificar perguntas em três níveis educacionais:
    
    - **Criança** (6-10 anos)
    - **Adolescente** (11-17 anos)  
    - **Adulto** (18+ anos)
    
    O modelo analisa a complexidade da pergunta e retorna uma resposta apropriada ao nível detectado.
    """
    )

    st.markdown("---")
    st.markdown("## 🎯 Exemplos de Perguntas")

    exemplos = {
        "👶 Criança": [
            "Por que o céu é azul?",
            "De onde vem a chuva?",
            "Como os bebês nascem?",
        ],
        "🧑 Adolescente": [
            "Como funciona a fotossíntese?",
            "O que é DNA?",
            "Como funciona a internet?",
        ],
        "👨‍🎓 Adulto": [
            "Qual o sentido da vida?",
            "Existe livre-arbítrio?",
            "O que é consciência?",
        ],
    }

    for nivel, perguntas in exemplos.items():
        st.markdown(f"**{nivel}**")
        for p in perguntas:
            if st.button(p, key=p, use_container_width=True):
                st.session_state.pergunta = p
                st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Recursos Avançados")


# Interface principal
st.markdown("---")

# Input da pergunta
pergunta = st.text_area(
    "📝 Digite sua pergunta:",
    height=100,
    placeholder="Ex: Qual o sentido da vida?",
    help="Digite qualquer pergunta e o sistema classificará o nível educacional apropriado e gerará uma resposta.",
)

# Botão de classificação
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classificar = st.button(
        "🔍 Classificar e Responder", type="primary", use_container_width=True
    )

if classificar and pergunta.strip():
    with st.spinner("Analisando a pergunta..."):
        try:
            # Classificar
            resultado = classifier(pergunta)[0]
            st.text(f"[DEBUG] Resultado bruto do classificador: {resultado}")
            label = resultado["label"]
            score = resultado["score"]

            # Extrair número do label (LABEL_0, LABEL_1, LABEL_2)
            nivel_num = int(label.split("_")[-1])
            nivel_nomes = {0: "Criança", 1: "Adolescente", 2: "Adulto"}
            nivel_nome = nivel_nomes[nivel_num]

            # Emojis por nível
            emojis = {0: "👶", 1: "🧑", 2: "👨‍🎓"}

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Classificação")
                st.markdown(f"### {emojis[nivel_num]} **Nível: {nivel_nome}**")
                st.metric(label="Confiança", value=f"{score*100:.2f}%", delta=None)

                # Barra de progresso
                st.progress(score)

                # Info adicional
                st.info(
                    f"Esta pergunta foi classificada como apropriada para o nível **{nivel_nome}** com {score*100:.1f}% de confiança."
                )

            with col2:
                st.markdown("### 💬 Resposta Apropriada")
                resposta = generate_answer(pergunta, nivel_num)

                # Box com cor de acordo com o nível
                if nivel_num == 0:
                    st.info(resposta)
                elif nivel_num == 1:
                    st.success(resposta)
                else:
                    st.warning(resposta)

                st.caption("🤖 Resposta gerada por IA")

            # Explicação dos níveis
            st.markdown("---")
            with st.expander("ℹ️ Entenda os níveis de classificação"):
                st.markdown(
                    """
                **👶 Criança (6-10 anos)**
                - Linguagem simples e direta
                - Comparações com situações do cotidiano
                - Explicações curtas e objetivas
                
                **🧑 Adolescente (11-17 anos)**
                - Conceitos mais elaborados
                - Termos técnicos introdutórios
                - Contexto científico e social
                
                **👨‍🎓 Adulto (18+ anos)**
                - Análise aprofundada
                - Referências acadêmicas e filosóficas
                - Múltiplas perspectivas e teorias
                """
                )

        except Exception as e:
            st.error(f"❌ Erro ao classificar: {str(e)}")

elif classificar and not pergunta.strip():
    st.warning("⚠️ Por favor, digite uma pergunta antes de classificar.")

# Rodapé
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Desenvolvido com Streamlit e Transformers 🤖</div>",
    unsafe_allow_html=True,
)

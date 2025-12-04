# Script para testar o modelo fine-tuned
from transformers import pipeline

print("=== Testando modelo ===")
print("Carregando modelo...")

classifier = pipeline("text-classification", model="./modelo_finetuned")
nivel_nomes = ["Criança", "Adolescente", "Adulto"]

textos_teste = [
    # Perguntas de criança
    "Por que o céu é azul?",
    "De onde vem a chuva?",
    "O que são estrelas?",
    "Por que precisamos dormir?",
    "Como os bebês nascem?",
    
    # Perguntas de adolescente
    "Como funciona a fotossíntese?",
    "O que é DNA?",
    "Como funciona a internet?",
    "O que causa terremotos?",
    "Como lidar com bullying na escola?",
    
    # Perguntas de adulto
    "Qual o sentido da vida?",
    "Existe livre-arbítrio?",
    "O que é consciência?",
    "Como a filosofia aborda a morte?",
    "Explique a teoria da relatividade",
    "Quais são as implicações éticas da inteligência artificial?",
]

print("\nTestando classificação de nível educacional:\n")

for texto in textos_teste:
    resultado = classifier(texto)[0]
    label_num = int(resultado['label'].split('_')[-1])
    print(f"Texto: {texto}")
    print(f"Nível: {nivel_nomes[label_num]} (confiança: {resultado['score']:.2%})")
    print("-" * 80)
    print()

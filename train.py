from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# Carregar dataset do arquivo JSONL
print("Carregando dataset...")
data = pd.read_json('dataset.jsonl', lines=True)
print(f"Total de exemplos: {len(data)}")
print(f"Colunas: {data.columns.tolist()}")
print(f"\nDistribuição de labels:")
print(data['label'].value_counts().sort_index())
print(f"\nPrimeiras linhas:")
print(data.head())

dataset = Dataset.from_pandas(data)

# Divide p/ treino/validação 80/20
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"\nExemplos de treino: {len(train_dataset)}")
print(f"Exemplos de validação: {len(eval_dataset)}")

print("\nTokenizando textos...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# Modelo - 3 classes (criança=0, adolescente=1, adulto=2)
print("\nCarregando modelo...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tuning
print("\nIniciando treinamento...")
trainer.train()
print("\nModelo treinado com sucesso!")

# Salvar modelo
trainer.save_model("./modelo_finetuned")
tokenizer.save_pretrained("./modelo_finetuned")
print("Modelo salvo em ./modelo_finetuned")
print("\nPara testar o modelo, execute: python test_model.py")
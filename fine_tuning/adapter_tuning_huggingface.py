"""

"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdapterConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

model_name = "distilbert-base-uncased"  # Choose any compatible model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add adapter
adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16)
model.add_adapter("my_adapter", config=adapter_config)
model.set_active_adapters("my_adapter")  # Set the added adapter as active

# Load a sample dataset
dataset = load_dataset("glue", "mrpc")  # MRPC dataset for demonstration

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

trainer.save_model("./adapter_model")
model.save_adapter("./adapter_model/my_adapter", "my_adapter")

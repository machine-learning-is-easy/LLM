from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from Lora import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For causal language models like GPT-2
    r=8,                           # Rank for low-rank decomposition
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Dropout for LoRA
    target_modules=["c_attn"]       # Target specific modules (GPT-2 uses 'c_attn' layers for attention)
)

# Apply LoRA configuration to the model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # This will show only the LoRA parameters are trainable
# Load dataset and tokenize
dataset = load_dataset("imdb", split="train")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,  # Use mixed precision training if possible
    save_total_limit=2,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
peft_model.save_pretrained("./peft_finetuned_model")
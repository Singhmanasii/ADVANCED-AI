

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define the padding token (usually the EOS token for GPT-2)
tokenizer.pad_token = tokenizer.eos_token  # This line is crucial to fix the error

# Load a text dataset (You can replace this with your own dataset)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Preprocess the dataset (Tokenization)
def encode(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=512)

# Tokenizing the dataset
train_data = dataset['train'].map(encode, batched=True)
val_data = dataset['validation'].map(encode, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_total_limit=2,
)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

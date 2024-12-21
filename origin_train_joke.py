"""
Original GPT-2 fine-tuning script - joke generation task

This script uses HuggingFace's Transformers library to fine-tune the pre-trained GPT-2 model.
The task is to generate joke text. Data loading uses text files.
The training process includes data preprocessing, training parameter setting, and model saving.
The data loading method is based on `TextDataset`, results in a fast training speed.

Workflow:
1. Load the pre-trained GPT-2 model and tokenizer.
2. Load the dataset and divide it into training and evaluation sets.
3. Implements static chunking on training and evaluation sets to ensure that the length of each sample meets the model requirements.
4. Define training parameters and fine-tune using HuggingFace's Trainer API.
5. Save the fine-tuned model and tokenizer.

Dependencies:
- torch
- sklearn
- transformers

Author: Jiaxin Tang, Jiayi Dong, Ximing Cai
Date: 2024/12/19
"""

import torch
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Check if the device supports GPU, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the special token to eos_token to avoid padding issues
tokenizer.pad_token = tokenizer.eos_token

# Load the entire dataset
with open("data/processed_jokes.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# Divide dataset into training and evaluation sets
train_texts, eval_texts = train_test_split(texts, test_size=0.1, random_state=42)

# Save training and evaluation sets to new files
with open("data/origin/train_jokes.txt", "w", encoding="utf-8") as f:
    f.writelines(train_texts)
    print(f"Raw text lines: {len(train_texts)}")
with open("data/origin/eval_jokes.txt", "w", encoding="utf-8") as f:
    f.writelines(eval_texts)
    print(f"Raw text lines: {len(eval_texts)}")

# Load training sets
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/origin/train_jokes.txt",
    block_size=128
)

# Load evaluation sets
eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/origin/eval_jokes.txt",
    block_size=128
)

print(f"train_dataset_size: {len(train_dataset)}, eval_dataset_size: {len(eval_dataset)}")

# Preprocessor, dynamically generating model inputs and labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 is an autoregressive model, no need MLM
)

# Set training parameters
training_args = TrainingArguments(
    output_dir="gpt2-jokes/origin",
    overwrite_output_dir=True,
    num_train_epochs=3, # 3 training rounds
    per_device_train_batch_size=4, # Training and evaluation batch size is both 4
    save_steps=3000,  # Save the model every 3000 steps
    eval_steps=3000,  # Evaluate every 3000 steps
    save_total_limit=2,  # Save 2 latest models
    logging_dir="logs/origin",
    logging_steps=50,  # Log every 50 steps
    logging_first_step=True,  # Log first step
    eval_strategy="steps",  # Evaluate model every few steps
    save_strategy="steps",  # Save model every few steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Use evaluation loss to select the best model
    greater_is_better=False,  # Smaller values in loss are better
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("gpt2-jokes/origin")
tokenizer.save_pretrained("gpt2-jokes/origin")
trainer.save_model("gpt2-jokes/origin")

print(f"Finish Training.")

"""
Modified GPT-2 fine-tuning script - joke generation task

This script uses HuggingFace's Transformers library to fine-tune the pre-trained GPT-2 model for the task of generating joke text.
Compared to the original script, it changes the way the dataset is loaded.
It leverages the HuggingFace "transformers" and "datasets" libraries to make it easier to dynamically process data.
This method of loading data results in a longer training time, but with better performance.

Workflow:
    1. Load a pre-trained GPT-2 model and tokenizer.
    2. Load the dataset and split it into training and evaluation sets.
    3. Define training parameters and fine-tune using HuggingFace's `Trainer` API.
    4. Save the fine-tuned model and tokenizer.

Dependencies:
    - torch
    - transformers
    - datasets

Author: Jiaxin Tang, Jiayi Dong, Ximing Cai
Date: 2024/12/20
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

# Check if the device supports GPU, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the special token to eos_token to avoid padding issues
tokenizer.pad_token = tokenizer.eos_token


def load_dataset(file_path, tokenizer, block_size=128):
    """
    Load and process the dataset from a text file.

    Args:
        file_path (str): Path to the text file containing the dataset.
        tokenizer (GPT2Tokenizer): The tokenizer used to encode the text.
        block_size (int, optional): Maximum length for the input sequence (default: 128).

    Returns:
        Dataset: HuggingFace Dataset object with tokenized inputs.
    """
    # Read all the lines in the file
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.readlines()

    # Tokenize, truncate and pad the text
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(encodings)
    return dataset


# Load dataset and encode
dataset = load_dataset("data/processed_jokes.txt", tokenizer)

# Divide dataset into training and evaluation set at a ratio of 90% and 10%
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]  # training set
eval_dataset = train_test_split["test"]  # evaluation set

print(f"train_dataset_size: {len(train_dataset)}, eval_dataset_size: {len(eval_dataset)}")

# Preprocessor, dynamically generating model inputs and labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 is an autoregressive model, no need MLM
)

# Set training parameters
training_args = TrainingArguments(
    output_dir="gpt2-jokes/modified",
    overwrite_output_dir=True,
    num_train_epochs=3,  # 3 training rounds
    per_device_train_batch_size=4,  # Training and evaluation batch size is both 4
    save_steps=3000,  # Save the model every 3000 steps
    eval_steps=3000,  # Evaluate every 3000 steps
    save_total_limit=2,  # Save 2 latest models
    logging_dir="logs/modified",
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

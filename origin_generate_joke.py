"""
Joke generation and evaluation script

The purpose of this script is to generate jokes using the fine-tuned GPT-2 model.
Then evaluate their quality based on various metrics such as perplexity, BERTScore, and distinct-n value.
The best joke is identified among the generated jokes based on the weighted average of these metrics.

Workflow:
1. Load the fine-tuned GPT-2 model and tokenizer.
2. Based on user-provided prompts, generate jokes.
3. Evaluate the jokes using with following metrics:
   - Perplexity (text fluency).
   - BERTScore (semantic similarity).
   - Distinct-n (lexical diversity).
   - Length of generated text.
4. Normalize the metrics and compute the weighted average to determine the best joke.

Command-line arguments:
    - `prompt`: Prompt for joke generation.
    - `--max_length`: Max length for the generated joke (default: 50).
    - `--num_return_sequences`: Number of generated jokes (default: 1).
    - `--temperature`: Sampling temperature (default: 0.7).

Dependencies:
    - re
    - torch
    - transformers
    - bert_score
    - argparse

Author: Jiaxin Tang, Jiayi Dong, Ximing Cai
Date: 2024/12/18
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import torch
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
from transformers import logging

# Check if the device supports GPU, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-jokes/origin"
print("Using Model Name:", model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Set the special token to eos_token to avoid padding issues
tokenizer.pad_token = tokenizer.eos_token


def generate_joke(prompt, max_length=50, num_return_sequences=1, temperature=0.7):
    """
    Generates jokes based on a user-provided prompts using the fine-tuned GPT-2 model.

    Args:
        prompt (str): The user-provided prompts.
        max_length (int, optional): Max length for the generated joke (default: 50).
        num_return_sequences (int, optional): Number of generated jokes (default: 1).
        temperature (float, optional): Sampling temperature (default: 0.7).

    Returns:
        list of str: A list of generated jokes.
    """
    # Encode input text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Call the generated function
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        no_repeat_ngram_size=2,  # Prevent duplicate ngram generation
    )

    # Decode generated text
    generated_texts = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
    # Extract the first sentence of each text and keep the punctuation
    sentences = []
    for text in generated_texts:
        # Matches ".", "?", "!", "\n", starting with the new generated text only
        match = re.match(r"([^.?!\n]*[.?!\n])", text[len(prompt):])
        if match:
            first_sentence = match.group(0)
        else:
            # If there is no punctuation, take the text directly
            first_sentence = text[len(prompt):]

        if len(first_sentence)==1:
            # Filter out text without content
            continue
        # If the first sentence does not have ".", "?", "!", add a "."
        if not any(punc in first_sentence for punc in ['.', '?', '!']):
            first_sentence += "."
        # Connect prompt to the first sentence
        full_text = prompt + " " + first_sentence.strip()
        sentences.append(full_text)
    return sentences


def calculate_perplexity(text):
    """
    Calculates the perplexity of input text using the fine-tuned GPT-2 model.

    Args:
        text (str): Input text.

    Returns:
        float: The perplexity value of the input text.
    """
    # Encode text into model input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    # Calculate model's loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross entropy loss
    # Perplexity is the exponential measure of cross entropy loss
    perplexity = torch.exp(loss).item()
    return perplexity


def calculate_bert_score(reference, generated_texts):
    """
    Calculates the BERTScore of the generated text when comparing to the reference.

    Args:
        reference (str): Reference text.
        generated_texts (str): Generated text.

    Returns:
        float: F1 score of the BERTScore.
    """
    # Compare only the generated text part, not the whole joke
    generated_part = generated_texts[len(reference):].strip()
    if isinstance(generated_part, str):
        # If generated text is a string, convert to a list
        generated_part = [generated_part]
    if isinstance(reference, str):
        # If reference text is a string, convert to a list
        reference = [reference]
    # Calculate BERTScore
    P, R, F1 = score(generated_part, reference, lang="en")
    # Return F1 score
    return F1.mean().item()


def distinct_n_grams(text, n=1):
    """
    Calculates the Distinct-n of a text.

    Args:
        text (str): Input text.
        n (int, optional): The size of the n-grams (default: 1).

    Returns:
        float: Distinct-n score (range: [0,1]).
    """
    tokens = text.split()
    n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(n_grams)) / len(n_grams) if len(n_grams) > 0 else 0


def normalize_perplexity(perplexity, min_val, max_val, epsilon=1e-5):
    """
    Normalizes the perplexity value to a range of [0, 1].

    Args:
        perplexity (float): Original perplexity value of a joke.
        min_val (float): The minimum perplexity value among generated jokes.
        max_val (float): The maximum perplexity value among generated jokes.
        epsilon (float, optional): A tiny value avoiding division by zero (default: 1e-5).

    Returns:
        float: The normalized perplexity value.
    """
    # Take the reciprocal of perplexity, min_val and max_val
    inverted_perplexity = 1 / perplexity if perplexity > 0 else 0
    inverted_min_val = 1 / min_val if min_val > 0 else 0
    inverted_max_val = 1 / max_val if max_val > 0 else 0
    # If inverted_min_val and inverted_max_val values are too close, return 0
    if inverted_min_val - inverted_max_val < epsilon:
        return 0
    # Normalize the reciprocal to [0,1]
    normalized_perplexity = (inverted_perplexity - inverted_max_val) / (inverted_min_val - inverted_max_val)
    return normalized_perplexity


def normalize_value(value, min_val, max_val, epsilon=1e-5):
    """
    Normalizes a metric value to [0, 1].

    Args:
        value (float): Original value to normalize.
        min_val (float): The minimum value among generated jokes.
        max_val (float): The maximum value among generated jokes.
        epsilon (float, optional): A tiny value avoiding division by zero (default: 1e-5).

    Returns:
        float: The normalized metric value.
    """
    if max_val - min_val < epsilon:
        return 0  # If max_val and min_val values are too close, return 0
    # Normalize to [0,1]
    normalized_value = (value - min_val) / (max_val - min_val)
    return normalized_value


def compute_combined_score(index, perplexity, bert_score, distinct_1, distinct_2, length):
    """
    Computes the combined score for a joke based on metrics.

    Args:
        index (int): Index of the joke.
        perplexity (list of float): List of perplexities.
        bert_score (list of float): List of BERTScores.
        distinct_1 (list of float): List of Distinct-1 values.
        distinct_2 (list of float): List of Distinct-2 values.
        length (list of int): List of lengths.

    Returns:
        float: Combined score for the joke.
    """
    weights = {'perplexity': 1.0, 'bert_score': 1.0, 'distinct_1': 0.5, 'distinct_2': 0.5, 'length': 1.0}
    # Normalizes each metric
    normalized_perplexity = normalize_perplexity(perplexity[index], min(perplexity), max(perplexity))
    normalized_bert_score = normalize_value(bert_score[index], min(bert_score), max(bert_score))
    normalized_distinct_1 = normalize_value(distinct_1[index], min(distinct_1), max(distinct_1))
    normalized_distinct_2 = normalize_value(distinct_2[index], min(distinct_2), max(distinct_2))
    normalized_length = normalize_value(length[index], min(length), max(length))
    print(f"Joke {index + 1} normalized_perplexity: {normalized_perplexity}")
    print(f"Joke {index + 1} normalized_bert_score: {normalized_bert_score}")
    print(f"Joke {index + 1} normalized_distinct_1: {normalized_distinct_1}")
    print(f"Joke {index + 1} normalized_distinct_2: {normalized_distinct_2}")
    print(f"Joke {index + 1} normalized_length: {normalized_length}")
    # Weighted average
    combined_score = (weights['perplexity'] * normalized_perplexity +
                      weights['bert_score'] * normalized_bert_score +
                      weights['distinct_1'] * normalized_distinct_1 +
                      weights['distinct_2'] * normalized_distinct_2 +
                      weights['length'] * normalized_length) / sum(weights.values())
    return combined_score


def main():
    """
    Main function for joke generation and evaluation.

    Parses command-line arguments to specify the input prompt and generation settings.
    Generates jokes using the fine-tuned GPT-2 model.
    Evaluates each generated joke based on perplexity, BERTScore, distinct-n and length.
    Computes combined score of each joke and outputs the best joke.
    """
    logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt for joke generation")
    parser.add_argument("--max_length", type=int, default=50, help="Max length for the generated joke")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of generated jokes")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Generate jokes and output
    generated_jokes = generate_joke(args.prompt, args.max_length, args.num_return_sequences, args.temperature)
    perplexities = []
    bert_scores = []
    distinct1s = []
    distinct2s = []
    lengths = []
    for i, joke in enumerate(generated_jokes):
        # Calculate perplexity
        perplexity = calculate_perplexity(joke)
        perplexities.append(perplexity)
        # Calculate BertScore
        bert_score = calculate_bert_score(args.prompt, joke)
        bert_scores.append(bert_score)
        # Calculate distinct-n
        distinct_1 = distinct_n_grams(joke[len(args.prompt):].strip(), n=1) # 不同的单词数与总单词数的比率
        distinct_2 = distinct_n_grams(joke[len(args.prompt):].strip(), n=2) # 不同的二元组数与总的二元组数的比率
        distinct1s.append(distinct_1)
        distinct2s.append(distinct_2)
        # Calculate length(token)
        length = len(joke[len(args.prompt):].strip())
        lengths.append(length)

        print(f"Generated Joke {i + 1}: {joke}")
        print(f"Joke {i + 1} Perplexity: {perplexity}")
        print(f"Joke {i + 1} BertScore: {bert_score}")
        print(f"Joke {i + 1} distinct_1: {distinct_1}")
        print(f"Joke {i + 1} distinct_2: {distinct_2}")
        print(f"Joke {i + 1} length: {length}")

    best_joke = None
    best_score = 0
    # Calculate combined score and find the best joke
    for i, joke in enumerate(generated_jokes):
        combined_score = compute_combined_score(i, perplexities, bert_scores, distinct1s, distinct2s, lengths)
        print(f"Joke {i + 1} Score: {combined_score}")
        # The higher the score, the better
        if combined_score >= best_score:
            best_score = combined_score
            best_joke = joke
    print(f"Best Joke: {best_joke}")


if __name__ == "__main__":
    main()

# COMP4132 Advanced Topics in Machine Learning Coursework

This implementation effectively integrates language model generation capabilities with multidimensional quality evaluation, providing an innovative solution for natural language generation tasks.

## 1 Installation Step

### 1.1 Clone Project

```shell
git clone https://github.com/DjyBrandon/atml-cw.git
```
or run local.

### 1.2 Install Dependencies

```shell
pip install -r requirements.txt
```

### 1.3 Run Program

Run program using [execute examples](#execute-examples) commands.

## 2 Execute Examples

### 2.1 Modified Model

```shell
python modified_generate_joke.py "How to split the sky into two parts?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "What's the difference between a human and a cat?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "What part of the chicken is chicken fillet?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "Who is faster in man and rabbit?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "Who flies higher, cats or people?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "What about the washing machine for cooking?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "What about eat pizza with chopsticks?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "How can I run faster than a plane?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python modified_generate_joke.py "What do you call a tomato in space?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```

### 2.2 Origin Model

```shell
python origin_generate_joke.py "How to split the sky into two parts?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "What's the difference between a human and a cat?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "What part of the chicken is chicken fillet?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "Who is faster in man and rabbit?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "Who flies higher, cats or people?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "What about the washing machine for cooking?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "What about eat pizza with chopsticks?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "How can I run faster than a plane?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```
```shell
python origin_generate_joke.py "What do you call a tomato in space?" --max_length 100 --num_return_sequences 10 --temperature 0.7
```

## 3 Project Structure

```text
atml-cw:
│  cached_lm_GPT2Tokenizer_128_processed_jokes.txt
│  command.txt
│  filter_joke.py
│  modified_generate_joke.py
│  modified_train_joke.py
│  origin_generate_joke.py
│  origin_train_joke.py
│  README.md
│  requirements.txt
│
├─data
│  │  processed_jokes.txt
│  │  sensitive_words.txt
│  │  shortjokes.csv
│  │  
│  └─origin
│          eval_jokes.txt
│          train_jokes.txt
│
├─gpt2-jokes
│  ├─modified
│  │  │  config.json
│  │  │  generation_config.json
│  │  │  merges.txt
│  │  │  model.safetensors
│  │  │  special_tokens_map.json
│  │  │  tokenizer_config.json
│  │  │  vocab.json
│  │  │  
│  │  ├─checkpoint-129000
│  │  │
│  │  └─checkpoint-130695
│  │
│  └─origin
│      │  config.json
│      │  generation_config.json
│      │  merges.txt
│      │  model.safetensors
│      │  special_tokens_map.json
│      │  tokenizer_config.json
│      │  vocab.json
│      │
│      ├─checkpoint-24000
│      │
│      └─checkpoint-24096
│
├─logs
│  ├─modified
│  │      events.out.tfevents.1734459492.Brandon.7537.0
│  │
│  └─origin
│          events.out.tfevents.1734494385.friedorange.10572.0
│
└─train-results
    ├─png
    │
    └─svg
```

## 4 Tensorboard Result

1. **Execute**: `tensorboard --logdir=logs`
2. **Access**: `http://localhost:6006/`

### 4.1 Train Results

<img src="train-results/svg/train_epoch.svg" width="50%"><img src="train-results/svg/train_grad_norm.svg" width="50%">

<img src="train-results/svg/train_learning_rate.svg" width="50%"><img src="train-results/svg/train_loss.svg" width="50%">

### 4.2 Eval Results

<img src="train-results/svg/eval_loss.svg" width="50%"><img src="train-results/svg/eval_runtime.svg" width="50%">

<img src="train-results/svg/eval_samples_per_second.svg" width="50%"><img src="train-results/svg/eval_steps_per_second.svg" width="50%">

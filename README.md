# NanoGPT Character-level Language Model

A minimal transformer-based GPT model trained on character-level data, inspired by [Andrej Karpathy’s nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy) and Attention Is All You Need ([Vaswani et al](https://arxiv.org/abs/1706.03762)). Built from scratch using PyTorch and customizable for different text datasets.

## About the Project

This project is an educational implementation of a GPT-style transformer model. I used this to learn the inner workings of transformer architectures, including self-attention, positional embeddings, and autoregressive text generation. Building this from scratch using PyTorch helped me understand each component of a GPT model and how it works, including capabilities and applications of a language model.

### Features

- Minimal transformer with self-attention and feedforward blocks  
- Character-level encoding/decoding
- Prompt-based generation using autoregressive sampling
- Easily configurable model depth, context length, and training hyperparameters  
- Dataset-agnostic: train on Shakespeare, tweets, books, or your own data  
- Runs on CPU or GPU (A100, L4, etc.)  
- Compatible with Colab for quick experimentation  

## Training

### Code
The code for the model along with basic training can be found in gpt.py. 

### Colab Notebook
More detailed hyperparameter tuning, training, and sample outputs can be found in this [notebook](https://colab.research.google.com/drive/18-S47xK6JYe_shhS9L3CM9SfFqnDt6Td?usp=sharing) on Google Colab.


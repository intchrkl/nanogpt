import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))


# Simple encoding for each letter (token)
char_to_int = { c:i for i, c in enumerate(chars)}
int_to_char = { i:c for i, c in enumerate(chars)}

# Encode a string S
def encode(S):
    return [char_to_int[c] for c in S]

# Decode a list of integers L
def decode(L):
    return ''.join([int_to_char[i] for i in L])

# Encoding the entire corpus and storing it in a Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Partition data into train and validation sets
percent_train = 0.9 # portion of data to be used as training set, rest is val
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

batch_size = 4 # no. of independent sequences to process in parallel
block_size = 8 # max no. of tokens to use for context in predictions

# Samples a mini-batch of sequence from either the training or validation set
def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idxs = torch.randint(0, len(data) - block_size, (batch_size,))
    input_tensors = torch.stack([data[i:i + block_size] for i in start_idxs])
    target_tensors = torch.stack([data[i + 1:i + block_size + 1] for i in start_idxs])
    return input_tensors.to(device), target_tensors.to(device)

# Sample input and target tensors:
# each row is an input to transformer
inputs, targets = get_batch('train')

def preview_batch():
    print("inputs:")
    print(inputs.shape)
    print(inputs)
    print("targets:")
    print(targets.shape)
    print(targets)
# preview_batch()

# Idea: take in n tokens as context and predict the (n + 1)th token
# Example:
#   Data: [24, 43, 58, 5, 57, 1, 46, 43]
#   Input tokens: [24], target: 43
#   Input tokens: [24, 43], target: 58
#   Input tokens: [24, 43, 58], target: 5
#   Input tokens: [24, 43, 58, 5], target: 57
#   Input tokens: [24, 43, 58, 5, 57], target: 1
#   Input tokens: [24, 43, 58, 5, 57, 1], target: 46
#   Input tokens: [24, 43, 58, 5, 57, 1, 46], target: 43

''' Bigram model: simplest 2-gram model that predicts each token based on 
    only the previous token. Will not be very accurate since this model only
    assumes that each token depends on only the one immediately before it.
    '''

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Using these embed dimensions so that the outputs can be raw logits 
        # directly used for the next token predictions
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # dim = (B, T, C) = (batch, time, channel)

        if targets is None:
            loss = None
        else:
            # PyTorch's cross_entropy expects (N, C) logits and (N,) targets — 
            # so we flatten across batch and time
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # How well we are predicting tokens based on logits
            loss = F.cross_entropy(logits, targets) 

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self.forward(idx)
            # Want to sample enxt token - so we get logits for only the last time step
            logits = logits[:, -1, :] # (B, C)
            # Convert logits into prob. distr. over vocab
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample one token from the prob. distr. and append to current sequence
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


vocab_size = len(chars)
model = BigramLM(vocab_size).to(device)
logits, loss = model(inputs, targets)
print(logits.shape)
print(loss)

# Start from a single token (= 0) and generate 100 new tokens
start = torch.zeros((1, 1), dtype=torch.long).to(device)
generated = model.generate(start, max_new_tokens=100)
print(decode(generated[0].tolist()))
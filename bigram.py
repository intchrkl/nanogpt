import torch
import torch.nn as nn
from torch.nn import functional as F

LINE_BREAK = "-----------------------------------\n"

torch.manual_seed(18916)
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

''' *** class BigramLM(nn.Module) ***
    Bigram model: simplest 2-gram model that predicts each token based on 
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
logits, loss = model.forward(inputs, targets)
print(logits.shape)
print(loss)

# Start from a single token (= 0) and generate 100 new tokens
start = torch.zeros((1, 1), dtype=torch.long).to(device)
generated = model.generate(start, max_new_tokens=100)
print("* Testing generation with empty prompt")
print("Input prompt:")
print("\'newline char\'")
print("Output:")
print(decode(generated[0].tolist()))
print(LINE_BREAK)

# Sample generated output - gibberish
'''lfJeukRuaRJKXAYtXzfJ:HEPiu--sDioi;ILCo3pHNTmDwJsfheKRxZCFs
lZJ XQc?:s:HEzEnXalEPklcPU cL'DpdLCafBheH'''

# Try generating with a starting prompt - still gibberish
prompt = "To be, or not to be"
prompt_idxs = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
generated = model.generate(prompt_idxs, max_new_tokens=100)
print("* Generating with a prompt")
print("Input prompt:")
print(f"\'{prompt}\'")
print("Output:")
print(decode(generated[0].tolist()))
print(LINE_BREAK)

# Training the model using a PyTorch optimizer
# The optimizer pdates model's params based on loss gradient to improve model performance
lr = 1e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
eval_interval = 32
max_iters = 10000
def train(verbose):
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch('train')
                _, loss = model.forward(xb, yb)
            if verbose: print(f"Step {iter}: loss = {loss.item():.4f}")
            model.train()

        # Sample batch of data
        xb, yb = get_batch('train')
        logit, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# Getting better but still bad
print(f"* Training with: max_iters = {max_iters}, eval_interval = {eval_interval}, lr={lr}")
train(verbose=False) # Change verbose to True to output loss during training
prompt = "To be, or not to be"
with_prompt = True
context = (torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device) 
            if with_prompt
            else torch.zeros((1, 1), dtype=torch.long).to(device))
max_new_tokens = 200
generated = model.generate(context, max_new_tokens=max_new_tokens)
print(decode(generated[0].tolist()))
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

LINE_BREAK = "-----------------------------------\n"

# hyperparameters
batch_size = 16 # no. of independent sequences to process in parallel
block_size = 32 # max no. of tokens to use for context in predictions
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(18916)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

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

# Samples a mini-batch of sequence from either the training or validation set
def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idxs = torch.randint(0, len(data) - block_size, (batch_size,))
    input_tensors = torch.stack([data[i:i + block_size] for i in start_idxs])
    target_tensors = torch.stack([data[i + 1:i + block_size + 1] for i in start_idxs])
    return input_tensors.to(device), target_tensors.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        loss_total = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            loss_total += loss.item()
        losses[split] = loss_total / eval_iters
    model.train()
    return losses

class Head(nn.Module):
    # A self-attention head

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # From Attention Is All You Need paper:
        wgt = q @ k.transpose(-2, -1) / math.sqrt(C) # QK^T/sqrt(d_k)
        wgt = wgt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wgt = F.softmax(wgt, dim=-1)
        wgt = self.dropout(wgt)

        v = self.value(x)
        out = wgt @ v
        return out

class MultiHeadAttention(nn.Module):
    # A collection of self-attention heads in parallel
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Multiple heads in parallel
        heads_list = [Head(head_size) for _ in range(num_heads)]
        self.heads = nn.ModuleList(heads_list)
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.self_attn = MultiHeadAttention(n_heads, head_size)
        self.feed_fwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.feed_fwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_final(x) 
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_and_context = idx[:, -block_size:]
            logits, _ = self.forward(idx_and_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate output from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_new_tokens = 1000
generated = model.generate(context, max_new_tokens=max_new_tokens)
print(decode(generated[0].tolist()))
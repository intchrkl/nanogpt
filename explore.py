# Exploring dataset - Tiny Shakespeare corpus

LINE_BREAK = "-----------------------------------\n"

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

print("* First 100 characters of the corpus:")
print(text[:100])
print(LINE_BREAK)
print(f"* Total corpus length: {len(text)}")
print(LINE_BREAK)
print("* Unique characters that occur in corpus:")
print(''.join(chars))
print(f"Unique characters: {len(chars)}")
print(LINE_BREAK)

# Simple encoding for each letter (token)
char_to_int = { c:i for i, c in enumerate(chars)}
int_to_char = { i:c for i, c in enumerate(chars)}

# Encode a string S
def encode(S):
    return [char_to_int[c] for c in S]

# Decode a list of integers L
def decode(L):
    return ''.join([int_to_char[i] for i in L])

print("* Simple encoding by enumerating characters in the corpus:")
msg = "Hello World"
print(encode(msg))
print(decode(encode(msg)))

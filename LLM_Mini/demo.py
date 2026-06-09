import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Importing custom Transformer block (attention + feed forward)
from transformer_blocks import Block


# Check PyTorch & GPU availability
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# Step 1: Define training corpus (raw text data)
corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is kantha",
    "the roads of Bangalore are busy",
    "it is raining in Kodathi",
    "the train is late again",
    "i love eating biryani and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "RCB won the IPL match"
]


# Step 2: Add END token → helps model learn sentence boundaries
corpus = [s + " <END>" for s in corpus]

# Combine all sentences into one string
text = " ".join(corpus)


# Step 3: Tokenization
# Split text into words and remove duplicates → vocabulary
words = list(set(text.split()))

# Vocabulary size
vocab_size = len(words)


# Step 4: Convert words → numbers (encoding)
# Model only understands numbers, not text
word2idx = {w: i for i, w in enumerate(words)}

# Reverse mapping → numbers back to words (for decoding output)
idx2word = {i: w for w, i in word2idx.items()}


# Step 5: Convert entire text into numerical tensor
data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)


# Step 6: Hyperparameters
block_size = 6      # How many words model looks at (context length)
embedding_dim = 32  # Size of word vectors
n_heads = 2         # Number of attention heads
n_layers = 2        # Number of transformer blocks
lr = 1e-3           # Learning rate
epochs = 1500       # Training iterations


# Step 7: Batch Processing
# Instead of training on full data → use mini-batches for efficiency
def get_batch(batch_size=16):

    # Random starting indices (ensures model sees different sequences)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # nput sequences (x)
    # Example: [I, love, tea]
    x = torch.stack([data[i:i+block_size] for i in ix])

    # Target sequences (y) → shifted by 1 word
    # Example: [love, tea, is]
    # Model learns: predict next word
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


# Step 8: Define TinyGPT Model
class TinyGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # Token embedding → converts word index to vectors
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Position embedding → adds word order information
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # Transformer blocks (attention layers)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )

        # Layer normalization for stability
        self.ln_f = nn.LayerNorm(embedding_dim)

        # Final linear layer → predicts next word probabilities
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):

        # idx shape → (Batch, Time)
        B, T = idx.shape

        # Convert word indices to embeddings
        tok_emb = self.token_embedding(idx)

        # Create positional embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        # Add token + position embeddings
        x = tok_emb + pos_emb

        # Pass through transformer (attention learning)
        x = self.blocks(x)

        # Normalize output
        x = self.ln_f(x)

        # Get logits (scores for each word)
        logits = self.head(x)

        loss = None

        # Compute loss if targets available
        if targets is not None:
            B, T, C = logits.shape

            # Flatten for cross entropy
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss


    # Step 9: Text Generation
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # Take last block_size tokens as context
            idx_cond = idx[:, -block_size:]

            # Predict next word
            logits, _ = self(idx_cond)

            # Take last time step output
            logits = logits[:, -1, :]

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next word (random selection)
            next_idx = torch.multinomial(probs, 1)

            # Append predicted word to sequence
            idx = torch.cat((idx, next_idx), dim=1)

        return idx


# Step 10: Initialize model & optimizer
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# Step 11: Training Loop
for step in range(epochs):

    # Get batch data
    xb, yb = get_batch()

    # Forward pass → prediction + loss
    logits, loss = model(xb, yb)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model weights
    optimizer.step()

    # Print loss every 300 steps
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


# Step 12: Text Generation
# Start with initial word "diwali"
context = torch.tensor([[word2idx["diwali"]]], dtype=torch.long)

# Generate new words
out = model.generate(context, max_new_tokens=15)

# Convert numbers back to text and print
print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))
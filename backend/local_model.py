import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Config (must match what you trained with) ─────────────────────────────────
BLOCK_SIZE = 256
N_EMBD     = 384
N_HEAD     = 6
N_LAYER    = 6
DROPOUT    = 0.0   # 0 at inference time
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'model.pt'
VOCAB_PATH = BASE_DIR / 'vocab.json'


# ── Tokenizer ─────────────────────────────────────────────────────────────────
class Tokenizer:
    def __init__(self, vocab_path: Path):
        with open(vocab_path) as f:
            data = json.load(f)
        self.stoi = data['stoi']
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.vocab_size = data['vocab_size']

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list) -> str:
        return ''.join(self.itos.get(i, '') for i in ids)


# ── Model Architecture (identical to training notebook) ───────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size)
        self.ff  = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Singleton loader ──────────────────────────────────────────────────────────
_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pt not found at {MODEL_PATH}. "
            "Copy your trained model.pt and vocab.json into the backend/ folder."
        )
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"vocab.json not found at {VOCAB_PATH}. "
            "Copy your vocab.json into the backend/ folder."
        )

    print(f"Loading AgentOS GPT from {MODEL_PATH}...")
    _tokenizer = Tokenizer(VOCAB_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    _model = GPT(vocab_size=_tokenizer.vocab_size).to(DEVICE)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()

    params = sum(p.numel() for p in _model.parameters())
    print(f"✅ Model loaded — {params/1e6:.1f}M parameters on {DEVICE}")
    print(f"   Val loss at training: {checkpoint.get('val_loss', 'N/A')}")

    return _model, _tokenizer


def generate(
    prompt: str = '',
    max_tokens: int = 400,
    temperature: float = 0.8,
    top_k: int = 40
) -> str:
    """
    Generate text from your trained model.
    Call this from agent.py to get responses.
    """
    model, tokenizer = load_model()

    if prompt:
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]
    else:
        ids = [0]

    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model.generate(ctx, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k)

    full_text = tokenizer.decode(out[0].tolist())

    # Return only the newly generated part if prompt was given
    if prompt and full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


def answer_question(question: str) -> str:
    """
    Format a question as a prompt and generate an answer.
    The model will complete the text — works best for factual/creative prompts.
    """
    # Format as Q&A to guide generation
    prompt = f"Question: {question}\nAnswer:"
    response = generate(prompt, max_tokens=300, temperature=0.7, top_k=30)
    # Clean up — take only the first coherent paragraph
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    return lines[0] if lines else response
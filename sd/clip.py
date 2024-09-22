import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        x += self.positional_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int) -> None:
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)

        residue = x

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        # Feedforward layer

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)
        
        # quickgelu 
        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x




class CLIP(nn.Module):
    def __init__(self):
        # vocab size, sequence length, padding 
        self.embeddings = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12,76) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)


        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)

        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)

        # (Batch_Size, Seq_Len, Dim)
        return output
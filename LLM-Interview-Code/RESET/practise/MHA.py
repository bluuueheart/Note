import torch
import torch.nn as nn
import torch.nn.functinal as F

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query, x_content=None, mask=None):
        batch_size = x_query.size(0)
        q = self.w_q(x_query)
        if x_content is None:
            k = self.w_k(x_query)
            v = self.w_v(x_query)
        else:
            k = self.w_k(x_content)
            v = self.w_v(x_content)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).tanspose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).tanspose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).tanspose(1, 2)
        scores = torch.matmul(q, k.tanspose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        content = context.transpose(1, 2)
        content = context.contiguous()
        output = context.view(batch_size, -1, self.model_dim)
        output = self.w_o(output)
        return output

if __name__ == "__main__":
    x = torch.randn(2, 10, 64)
    mha = MultiHeadAttention(model_dim=64, num_heads=8)
    out = mha(x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")


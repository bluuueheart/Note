import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # 每个头的维度

        self.w_q = nn.Linear(model_dim, model_dim)  # 输入映射为 Q, K, V投影层
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

        self.w_o = nn.Linear(model_dim, model_dim)  # 输出投影层
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query, x_context=None, mask=None):
        batch_size = x_query.size(0)

        q = self.w_q(x_query)
        if x_context is not None:   # Self-Attention 和 Cross-Attention
            k = self.w_k(x_context)
            v = self.w_v(x_context)
        else:
            k = self.w_k(x_query)
            v = self.w_v(x_query)

        # 分头处理：[batch_size, seq_len, model_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # 转置k(交换后2维), 矩阵乘法q @ k^T

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # 在最后1维上做 softmax，对所有 key 位置归一化
        attn_weights = self.dropout(attn_weights) # 对注意力权重做 dropout, 训练时随机丢掉一部分注意力连接, 防止过拟合

        context = torch.matmul(attn_weights, v) # 用注意力权重加权 V

        # ========== 合并多头 ==========
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads, head_dim] -> [batch_size, seq_len_q, model_dim]
        context = context.transpose(1, 2)
        context = context.contiguous()  # transpose 后，tensor 的内存通常不是连续的
        output = context.view(batch_size, -1, self.model_dim) # reshape context
        
        output = self.w_o(output)   # 输出投影
        return output
    
if __name__ == "__main__":
    # batch_size=2, seq_len=10, model_dim=64, num_heads=8
    x = torch.randn(2, 10, 64)
    mha = MultiHeadAttention(model_dim=64, num_heads=8)
    out = mha(x, x)  # Self-Attention: x_query=x, x_context=x
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # 应该还是 (2, 10, 64)
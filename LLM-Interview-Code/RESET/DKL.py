def kl_divergence(p, q):
    """
    计算两个离散分布的KL散度
    KL(P || Q) = Σ p_i * log(p_i / q_i)
    :param p: 真实概率分布，形状 (n_classes, )
    :param q: 预测概率分布，形状 (n_classes, )
    :return: 标量散度值
    """
    # 过滤零元素避免数值问题
    mask = (p != 0)
    p = p[mask]
    q = q[mask]
    return np.sum(p * np.log(p / q))

# 示例用法（概率分布差异对比）
P = np.array([0.4, 0.6])
Q1 = np.array([0.4, 0.6])
Q2 = np.array([0.5, 0.5])

print(f"KL(P||Q1): {kl_divergence(P, Q1):.4f}")  # 输出 0.0000
print(f"KL(P||Q2): {kl_divergence(P, Q2):.4f}")  # 输出 0.0204
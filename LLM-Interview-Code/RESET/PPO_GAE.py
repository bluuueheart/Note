import torch
import torch.nn as nn
import torch.optim as optim
 
def ppo_loss(probs, old_probs, advantages, actions, clip_epsilon=0.2):
    """
    PPO 策略损失（裁剪目标）
    Args:
        probs: 当前策略的动作概率分布 (batch_size, num_actions)
        old_probs: 旧策略的动作概率分布 (batch_size, num_actions)
        advantages: 优势估计 (batch_size,)
        actions: 实际执行的动作索引 (batch_size,)
        clip_epsilon: 裁剪范围
    Returns:
        loss: 标量损失
    """
    # 取出对应动作的概率
    prob = probs.gather(1, actions.unsqueeze(1)).squeeze(1)   # (batch_size,)
    old_prob = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    # 概率比 r_t
    ratio = prob / (old_prob + 1e-8)
    # 裁剪后的比值
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    # 原始目标和裁剪目标
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    # 取最小值并取负号（因为我们要最小化损失）
    loss = -torch.min(surr1, surr2).mean()
    return loss
 
def value_loss(values, returns):
    """
    价值网络损失（MSE）
    Args:
        values: 状态价值估计 (batch_size,)
        returns: 实际回报 (batch_size,)
    """
    return nn.MSELoss()(values, returns)
 
def entropy_loss(probs):
    """
    熵奖励（负熵，用于最大化熵）
    Args:
        probs: 动作概率分布 (batch_size, num_actions)
    Returns:
        负熵的均值（损失中减去熵奖励，所以此处返回 -entropy）
    """
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    return -entropy   # 损失中会减去熵，因此返回负熵
 
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    计算 GAE 优势
    Args:
        rewards: 奖励序列列表 (list of length T)
        values: 价值估计序列 (list of length T+1, 最后一个为 V(s_{T+1}))
        gamma: 折扣因子
        lam: GAE lambda 参数
    Returns:
        advantages: 优势序列 (list of length T)
        returns: 回报序列 (list of length T)
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    # 计算回报：return = advantage + value
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns
 
# ===== 伪代码框架 =====
def ppo_train_step(policy_net, value_net, optimizer_p, optimizer_v, trajectories, epochs=10, clip_epsilon=0.2):
    """
    单次 PPO 更新步骤（伪代码）
    trajectories: 收集的一批轨迹，每个轨迹包含 states, actions, rewards, old_probs, old_values
    """
    # 预先计算 GAE 和回报
    all_advantages = []
    all_returns = []
    for traj in trajectories:
        # traj: states, actions, rewards, old_probs, old_values (list of values at each step)
        # 注意 old_values 长度比 rewards 多 1，最后一个是 V(s_{T+1})
        adv, ret = compute_gae(traj['rewards'], traj['values'], gamma=0.99, lam=0.95)
        all_advantages.extend(adv)
        all_returns.extend(ret)
    all_advantages = torch.tensor(all_advantages)
    all_returns = torch.tensor(all_returns)
    # 将轨迹数据打包成 batch
    # ...（此处省略数据准备）
    for _ in range(epochs):
        # 随机采样 mini-batch
        for batch in dataloader:
            states, actions, old_probs, advantages, returns = batch
            # 前向传播
            new_probs = policy_net(states)  # 输出分布
            new_values = value_net(states).squeeze()
            # 计算损失
            loss_p = ppo_loss(new_probs, old_probs, advantages, actions, clip_epsilon)
            loss_v = value_loss(new_values, returns)
            entropy = entropy_loss(new_probs)
            total_loss = -loss_p + 0.5 * loss_v - 0.01 * entropy
            # 反向传播
            optimizer_p.zero_grad()
            optimizer_v.zero_grad()
            total_loss.backward()
            optimizer_p.step()
            optimizer_v.step()
        pass
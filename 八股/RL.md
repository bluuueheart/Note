# 后训练（主要是RL）
```
大模型后训练
│
├── SFT
│
├── RLHF
│   ├── SFT
│   ├── Reward Model
│   └── RL Optimization
│       ├── PPO
│       ├── A2C
│       ├── GRPO
│       └── 其他 RL 算法
│
├── DPO
│
├── IPO / KTO / ORPO
│
└── RLVR
    ├── PPO
    ├── GRPO
    └── DAPO
```

## 1. 强化学习基础

### 1.1 一句话定义

**强化学习 Reinforcement Learning, RL** 是让智能体 `agent` 在环境 `environment` 中通过试错选择动作 `action`，最大化长期累计奖励 `return` 的学习范式。

面试一句话：

> 强化学习的核心是学习一个策略 $\pi_\theta(a|s)$，使得智能体在状态 $s$ 下选择动作 $a$ 后获得的长期折扣奖励最大。

---

### 1.2 MDP 基本形式

强化学习通常建模为马尔可夫决策过程：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

其中：

| 符号            | 含义    |        |
| ------------- | ----- | ------ |
| $\mathcal{S}$ | 状态空间  |        |
| $\mathcal{A}$ | 动作空间  |        |
| $P(s'   s,a)$ | 状态转移概率 |
| $R(s,a)$      | 奖励函数  |        |
| $\gamma$      | 折扣因子，控制模型看重短期/长期奖励  |        |

目标是最大化期望回报：

$$
J(\theta)=\mathbb{E}*{\tau \sim \pi*\theta}
\left[
\sum_{t=0}^{T}\gamma^t r_t
\right]
$$

其中轨迹：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\cdots)
$$

---

### 1.3 LLM / VLM 中的 RL 如何对应 MDP

在大模型后训练里，可以这样对应：

| 传统 RL           | LLM / VLM RL                                        |
| --------------- | --------------------------------------------------- |
| 状态 $s_t$        | prompt + 已生成 token                                  |
| 动作 $a_t$        | 下一个 token / tool call                               |
| 策略 $\pi_\theta$ | 当前语言模型                                              |
| 环境              | 生成过程、外部工具、评测器                                       |
| 奖励 $r$          | reward model / rule-based reward / human preference |
| episode         | 一次完整回答生成                                            |

对于 LLM：

$$
a_t = y_t
$$

也就是每一步动作就是生成下一个 token。

策略概率为：

$$
\pi_\theta(y_t|x,y_{<t})
$$

整段回答概率为：

$$
\pi_\theta(y|x)=\prod_{t=1}^{|y|}
\pi_\theta(y_t|x,y_{<t})
$$

---

## 2. 强化学习算法发展历程

### 2.1 总体脉络

| 阶段              | 代表方法                   | 核心思想                          |
| --------------- | ---------------------- | ----------------------------- |
| 传统 RL           | Q-learning, SARSA      | 学习状态-动作价值                     |
| 深度 RL           | DQN, A3C               | 用神经网络近似 value / policy        |
| Policy Gradient | REINFORCE              | 直接优化策略                        |
| Actor-Critic    | A2C, A3C, PPO          | policy + value function       |
| LLM 对齐          | RLHF + PPO             | 人类偏好训练 reward model，再用 PPO 优化 |
| 离线偏好优化          | DPO / IPO / KTO / ORPO | 不做在线 RL，直接用偏好对优化              |
| 推理 RL           | GRPO / RLVR            | 用可验证 reward 激发 reasoning      |
| 长 CoT RL        | DAPO / Dr.GRPO 等       | 解决 GRPO 在长推理中的熵塌陷、无效样本、长度偏置   |

RLHF 在 InstructGPT 中被系统化使用：先 SFT，再训练 reward model，最后用 PPO 优化策略，使语言模型更符合人类偏好。([arXiv][1]) DPO 则提出把偏好优化转成一个简单分类损失，不需要显式 reward model，也不需要在线采样式 RL，因而实现更简单、更稳定。([arXiv][2]) GRPO 最早在 DeepSeekMath 中提出，是 PPO 的变体，通过 group-relative reward 替代 critic，从而降低 PPO 的内存占用。([arXiv][3]) DAPO 进一步针对长 CoT RL 提出 Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping 四个关键技巧。([arXiv][4])

---

## 3. 强化学习基本步骤和流程

### 3.1 传统 RL 流程

```text
初始化策略 πθ
for iteration in range(N):
    1. 用当前策略与环境交互，采样轨迹 τ
    2. 根据环境反馈计算 reward
    3. 估计 return / advantage
    4. 用 policy gradient 更新策略
    5. 评估策略，继续迭代 
```

核心梯度：

$$
\nabla_\theta J(\theta)
=

\mathbb{E}*{\tau \sim \pi*\theta}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) A_t
\right]
$$

其中 $A_t$ 是 advantage：

$$
A_t = Q(s_t,a_t)-V(s_t)
$$

直观理解：

> 如果一个动作带来的结果比平均水平好，就提高它的概率；如果比平均水平差，就降低它的概率。

---

### 3.2 LLM RL 流程

```text
输入 prompt batch
    ↓
Actor 模型生成 responses
    ↓
Reward model / rule-based reward 打分
    ↓
计算 advantage
    ↓
PPO / GRPO / DAPO 更新 actor
    ↓
同步 old policy / reference policy
    ↓
继续 rollout
```

LLM RL 的一次训练 step 通常包括：

1. **Rollout generation**：模型生成回答。一次rollout通常包含多个 prompt，每个 prompt 采样多个回答。
2. **Reward computation**：用奖励模型、规则、评测器打分。
3. **Advantage estimation**：计算每个回答或 token 的 advantage。
4. **Policy optimization**：更新当前模型。
5. **KL control**：防止模型偏离 reference model 太远。衡量新旧策略差异。
6. **Evaluation**：看 reward、KL、entropy、length、accuracy 等指标。

---

## 4. RLHF + PPO

### 4.1 RLHF 是什么

RLHF，全称 **Reinforcement Learning from Human Feedback**，核心是把人类偏好蒸馏成 reward model，然后用强化学习优化语言模型。

经典流程：

```text
Pretrained LM
    ↓
SFT：用人工高质量答案监督微调
    ↓
Reward Model：用人类偏好对训练奖励模型
    ↓
PPO：用 reward model 作为奖励优化 policy
```
InstructGPT 的 RLHF 流程正是这种三阶段范式。([arXiv][1])

---

### 4.2 Reward Model 训练

给定同一个 prompt $x$，有两个回答：

* $y_w$：winner，人类更喜欢的回答；
* $y_l$：loser，人类不喜欢的回答。

Reward model 输出：

$$
r_\phi(x,y)
$$

希望：

$$
r_\phi(x,y_w) > r_\phi(x,y_l)
$$

常用 Bradley-Terry loss：

$$
\mathcal{L}_{RM}
=

-\log \sigma
\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
$$

---

### 4.3 PPO 基本原理

PPO，全称 **Proximal Policy Optimization**，核心是限制新旧策略变化幅度，避免 policy update 太激进。
下面讲的是PPO-CLIP

定义概率比值：

$$
\rho_t(\theta)
=

\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

PPO clipped objective：

$$
\mathcal{L}^{CLIP}(\theta)
=

\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta) A_t,
\text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

* $\pi$是策略，$a_t$是动作，$s_t$是状态。
* $\rho_t(\theta)$是重要性权重
* 需要最大化 $\mathcal{L}^{CLIP}(\theta)$ (不同于主流优化器如 AdamW 会最小化负的 loss)
* clip函数：$\text{clip}(x, l, r) = \min(\max(x, l), r)$ x在[l, r]区间内不变，超出区间则被截断到边界。

含义：

* 如果新策略比旧策略概率大太多，clip 截断；
* 如果新策略比旧策略概率小太多，也 clip 截断；
* 防止模型一次更新过猛导致崩掉。

---

### 4.4 LLM-RLHF 中 PPO 的目标

LLM 中一般还有 KL 惩罚：

$$
\hat r(x,y)
=
r_\phi(x,y)

\beta
D_{KL}
\left[
\pi_\theta(\cdot|x)
|
\pi_{\text{ref}}(\cdot|x)
\right]
$$

也就是：

> reward model 鼓励更符合人类偏好的回答，KL 惩罚防止模型偏离原始模型太远。

---

### 4.5 PPO 回答模板

> PPO 是一种 on-policy actor-critic 强化学习算法。它通过新旧策略概率比值构造 surrogate objective，并用 clip 限制策略更新幅度。在 RLHF 中，PPO 通常用于优化语言模型，使模型最大化 reward model 给出的奖励，同时通过 KL 散度 约束模型不要偏离 reference model 太远。PPO 的缺点是资源开销大，因为通常需要 actor、critic、reference model、reward model 多个模型参与训练。

Actor-Critic 是强化学习中结合策略学习和价值学习的方法。Actor 学习策略 $\pi(a|s)$，负责选择动作；Critic 学习价值函数 $V(s)$ 或 $Q(s,a)$，负责评价动作好坏。Critic 计算 advantage，Actor 根据 advantage 调整动作概率。PPO 是典型 Actor-Critic；GRPO 则去掉 Critic，用 group 内 reward 归一化估计 advantage。
除了 Actor-Critic，RL 还有 value-based(不直接学习策略，而是学习每个动作有多好，然后选价值最高的动作 不适合超大动作空间)、policy-based(直接学习策略 $\pi_\theta(a|s)$，让好动作概率变大，坏动作概率变小)、model-based、model-free、on-policy、off-policy、offline RL、imitation learning(不通过 reward 学，而是模仿专家行为) 和 preference optimization 等类型。
actor-critic 流程：
```
初始化 Actor πθ 和 Critic Vφ
for each iteration:
    1. Actor 根据当前状态 s_t 采样动作 a_t
    2. 环境返回 reward r_t 和下一个状态 s_{t+1}
    3. Critic 估计 V(s_t), V(s_{t+1})
    4. 计算 TD error 或 advantage
    5. 用 advantage 更新 Actor
    6. 用 value loss 更新 Critic
```
---

## 5. DPO

### 5.1 DPO 是什么

DPO，全称 **Direct Preference Optimization**。

它不是传统意义上的在线 RL，而是一种 **离线偏好优化算法**。

核心思想：

> 不再显式训练 reward model，也不再用 PPO 做在线 rollout，而是直接用 chosen / rejected 偏好对优化语言模型。

DPO 论文指出，RLHF 需要先训练 reward model 再用 RL 优化，流程复杂且不稳定；DPO 通过重新参数化 reward，将偏好学习转成简单的分类损失。([arXiv][2])

### 5.2 DPO 公式

给定 prompt $x$，chosen answer $y_w$，rejected answer $y_l$。

定义隐式 reward：

$$
r_\theta(x,y)
=

\beta
\log
\frac{
\pi_\theta(y|x)
}{
\pi_{\text{ref}}(y|x)
}
+
\beta \log Z(x)
$$

DPO loss：

$$
\mathcal{L}_{DPO}
=

-\mathbb{E}*{(x,y_w,y_l)}
\left[
\log \sigma
\left(
\beta
\left[
\log \frac{\pi*\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
\right]
$$

直观理解：

> DPO 希望当前模型相对 reference model 更提高 chosen 的概率，同时更降低 rejected 的概率。

### 5.3 DPO 代码骨架

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_log_ratio - rejected_log_ratio)

    loss = -F.logsigmoid(logits).mean()
    return loss
```

### 5.4 DPO 优缺点

| 方面    | DPO                                          |
| ----- | -------------------------------------------- |
| 优点    | 简单、稳定、不需要 reward model、不需要在线 rollout         |
| 缺点    | 依赖高质量偏好数据；不能像 RL 那样在线探索；对复杂可验证推理任务不如 RLVR 灵活 |
| 适合场景  | 对齐、风格偏好、安全偏好、回答质量偏好                          |
| 不适合场景 | 需要模型自己探索长 CoT、数学、代码、多步工具调用                   |

---

## 6. GRPO

### 6.1 GRPO 是什么

GRPO，全称 **Group Relative Policy Optimization**。

它是 DeepSeekMath 提出的 PPO 变体，后续在 DeepSeek-R1 系列推理 RL 中被广泛讨论。DeepSeekMath 把 GRPO 描述为 PPO 变体，用 group-relative advantage 替代 value model，从而降低内存消耗。([arXiv][3]) DeepSeek-R1 则使用大规模 RL 激发推理能力，R1-Zero 不经过 SFT 直接 RL，R1 则加入 cold-start data 和多阶段训练以改善可读性与语言混杂问题。([arXiv][5])

### 6.2 GRPO 核心思想

PPO 需要 critic / value model：

$$
A_t = Q(s_t,a_t)-V(s_t)
$$

GRPO 不训练 value model，而是：

> 对同一个 prompt 采样一组回答，用这一组回答内部的 reward 均值和方差做相对归一化。

给定 prompt $x$，采样 $G$ 个回答：

$$
{y_1,y_2,\cdots,y_G}
$$

每个回答 reward：

$$
r_i = R(x,y_i)
$$

group mean：

$$
\mu = \frac{1}{G}\sum_{i=1}^{G} r_i
$$

group std：

$$
\sigma = \sqrt{
\frac{1}{G}
\sum_{i=1}^{G}
(r_i-\mu)^2
}
$$

advantage：

$$
A_i
=

\frac{r_i-\mu}{\sigma+\epsilon}
$$

### 6.3 GRPO 损失

token-level ratio：

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t}|x,y_{i,<t})
}{
\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})
}
$$

GRPO objective：

$$
\mathcal{L}_{GRPO}
=

*

\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}
\left[
\min
\left(
\rho_{i,t} A_i,
\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
-

\beta D_{KL}
(\pi_\theta | \pi_{\text{ref}})
\right]
$$

很多最新 reasoning RL 实现会减弱甚至移除 KL 项。TRL 文档也提到，近期 GRPO 实践中常把 `beta=0.0` 作为默认，认为 KL 不是必需项。([Hugging Face][6])

### 6.4 GRPO 为什么省资源

PPO 通常需要：

```text
actor model
critic / value model
reward model
reference model
old policy
```

GRPO 去掉 critic：

```text
actor model
reference model
old policy
reward function / reward model
```

所以省掉：

* value model 参数；
* value model optimizer state；
* value forward；
* GAE 计算；
* critic 训练不稳定问题。

但 GRPO 也有新的开销：

> 每个 prompt 要采样 $G$ 个回答，所以 rollout 开销会显著增加。

---

### 6.5 GRPO 代码骨架

```python
import torch

def compute_grpo_advantage(rewards, eps=1e-8):
    """
    rewards: shape [batch_size, group_size]
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)

    advantages = (rewards - mean) / (std + eps)

    # std 接近 0 说明 group 内 reward 全一样，基本没有有效学习信号
    valid_mask = std.squeeze(1) > eps

    return advantages, valid_mask


rewards = torch.tensor([
    [1., 0., 1., 0.],  # 有区分度
    [1., 1., 1., 1.],  # 全对，无梯度
    [0., 0., 0., 0.],  # 全错，无梯度
])

adv, valid = compute_grpo_advantage(rewards)

print(adv)
print(valid)
```

---

## 7. 追问：GRPO 中一个 group 全是 0 或全是 1，怎么办？

### 7.1 问题本质

如果同一个 prompt 的 $G$ 个回答 reward 全一样：

$$
r_1=r_2=\cdots=r_G
$$

则：

$$
\sigma = 0
$$

advantage：

$$
A_i
===

# \frac{r_i-\mu}{\sigma+\epsilon}

0
$$

也就是说：

```text
全 1：模型已经都会了，组内没有谁比谁更好
全 0：模型全不会，组内也没有谁比谁更好
```

结果：

> advantage 全为 0，policy gradient 近似为 0，这个 prompt 对训练没有贡献。

DAPO 论文明确指出，当某些 prompt 的 group accuracy 等于 1 时，GRPO 会出现 zero advantage，从而没有 policy update 梯度；DAPO 的 Dynamic Sampling 会过滤掉 accuracy 为 0 或 1 的 group，只保留有有效梯度的样本。([arXiv][4])

---

### 7.2 解决方法一：DAPO Dynamic Sampling

定义 group accuracy：

$$
acc(x)
======

\frac{1}{G}
\sum_{i=1}^{G}
\mathbb{1}[R(x,y_i)=1]
$$

有效样本条件：

$$
0 < acc(x) < 1
$$

也就是：

```text
至少有一个答对
并且至少有一个答错
```

保留这种 prompt，因为它能提供相对比较信号。

过滤条件：

$$
\sum_{i=1}^{G} r_i \neq 0
\quad \text{and} \quad
\sum_{i=1}^{G} r_i \neq G
$$

---

### 7.3 Dynamic Sampling 伪代码

```python
def dynamic_sampling(policy, prompts, reward_fn, group_size, target_batch_size):
    buffer = []

    while len(buffer) < target_batch_size:
        batch_prompts = sample_prompts(prompts)

        # 每个 prompt 采样 G 个回答
        responses = policy.generate(
            batch_prompts,
            num_return_sequences=group_size,
            temperature=1.0,
        )

        rewards = reward_fn(batch_prompts, responses)
        # rewards: [batch_size, group_size], values in {0, 1}

        group_correct = rewards.sum(dim=1)

        # 只保留非全对、非全错的 group
        valid = (group_correct > 0) & (group_correct < group_size)

        for prompt, resp_group, reward_group in zip(
            batch_prompts[valid],
            responses[valid],
            rewards[valid],
        ):
            buffer.append((prompt, resp_group, reward_group))

    return buffer[:target_batch_size]
```

---

### 7.4 解决方法二：调整采样温度和 group size

如果全 1 太多：

```text
说明题太简单，或者模型已经学会。
```

可以：

* 提高题目难度；
* 提高采样温度；
* 增大 group size；
* 引入 hard prompt mining；
* 降低 easy prompt 采样概率。

如果全 0 太多：

```text
说明题太难，模型完全探索不到正确解。
```

可以：

* 降低题目难度；
* 使用 curriculum learning；
* 加 cold-start SFT；
* 提高 rollout 数量；
* 引入 process reward；
* 引入 partial reward；
* 加 verifier / tool feedback。

---

### 7.5 解决方法三：使用 partial reward / dense reward

对于数学、代码、视频理解等任务，只有 final answer reward 太稀疏。

可以设计：

$$
R = R_{\text{answer}} + \lambda R_{\text{format}} + \alpha R_{\text{process}}
$$

例如：

| reward          | 含义       |
| --------------- | -------- |
| answer reward   | 最终答案是否正确 |
| format reward   | 是否满足格式   |
| process reward  | 中间推理是否合理 |
| tool reward     | 工具调用是否有效 |
| evidence reward | 是否找到正确证据 |

---

### 7.6 面试回答模板

> GRPO 的 advantage 是在同一个 prompt 的 group 内做 reward 标准化。如果 group 内 reward 全是 0 或全是 1，那么标准差为 0，所有 response 的相对 advantage 都是 0，导致这个 prompt 没有有效梯度。解决方法是 DAPO 里的 Dynamic Sampling：对 prompt 过采样，然后过滤掉 group accuracy 为 0 或 1 的样本，只保留 $0<acc<1$ 的 group，使 batch 中每个 prompt 都有相对偏好信号。对于全 0 问题，还可以用 curriculum、cold-start、partial reward 或 process reward 增强探索；对于全 1 问题，可以提高题目难度或降低 easy prompt 采样权重。

---

## 8. DAPO

### 8.1 DAPO 是什么

DAPO，全称：

```text
Decoupled Clip and Dynamic sAmpling Policy Optimization
```

DAPO 论文提出四个关键技巧：

1. **Clip-Higher**
2. **Dynamic Sampling**
3. **Token-Level Policy Gradient Loss**
4. **Overlong Reward Shaping**

DAPO 官方实验在 Qwen2.5-32B base 上做长 CoT RL，目标是解决 naive GRPO 的 entropy collapse、reward noise、training instability 等问题。([arXiv][4])

---

### 8.2 Clip-Higher

PPO / GRPO 通常使用对称 clip：

$$
\text{clip}(\rho,1-\epsilon,1+\epsilon)
$$

DAPO 改成非对称 clip：

$$
\text{clip}(\rho,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})
$$

其中：

$$
\epsilon_{\text{high}} > \epsilon_{\text{low}}
$$

作用：

> 给低概率 token 更大的上升空间，缓解 entropy collapse，提高探索能力。

DAPO 论文指出，过低的 upper clip 会限制低概率 exploration token 的概率提升，因此 Clip-Higher 提高上界以增强多样性。([arXiv][4])

---

### 8.3 Dynamic Sampling

如上节所说：

$$
0 < acc(x) < 1
$$

只保留有区分度的 prompt group。

DAPO 论文的 progressive results 显示，Dynamic Sampling 是提升最大的技巧之一；其结果表中 naive GRPO 为 30，加入各技巧后 DAPO 达到 50。([arXiv][4])

---

### 8.4 Token-Level Policy Gradient Loss

原始 GRPO 常常先对每个 sample 内 token loss 求平均，再对 sample 求平均。

问题：

> 长回答和短回答在 sample 级别权重一样，导致长 CoT 中每个 token 的梯度权重被稀释。

DAPO 改成 token-level loss，让所有 token 更公平地参与优化。

样本级 loss 类似：

$$
\mathcal{L}_{sample}
====================

-\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}
l_{i,t}
$$

DAPO token-level loss 类似：

$$
\mathcal{L}_{token}
===================

*

\frac{
\sum_{i=1}^{G}
\sum_{t=1}^{|y_i|}
l_{i,t}
}{
\sum_{i=1}^{G}
|y_i|
}
$$

DAPO 论文指出，在长 CoT 场景中，sample-level loss 会让长回答中的 token 贡献偏低，Token-Level Policy Gradient Loss 可以改善长推理训练稳定性。([arXiv][4])

---

### 8.5 Overlong Reward Shaping

长 CoT 训练中，回答可能超长并被截断。

如果简单地把所有截断样本都判负，会有 reward noise：

```text
模型可能推理过程本身是对的，只是太长被截断。
```

DAPO 的思路：

* 对 overlong response 做 soft punishment；
* 不要简单粗暴全判错；
* 在长度接近上限时逐渐惩罚；
* 减少 reward noise，稳定训练。

DAPO 论文指出，不恰当地惩罚 truncated samples 会引入 reward noise；Overlong Reward Shaping / Filtering 能稳定训练。([arXiv][4])

---

## 9. PPO / DPO / GRPO / DAPO 对比

| 方法       | 是否在线采样 | 是否需要 reward model | 是否需要 critic | 核心优势                       | 核心缺点                |
| -------- | -----: | ----------------: | ----------: | -------------------------- | ------------------- |
| PPO-RLHF |      是 |              通常需要 |          需要 | 经典稳定，适合复杂 reward           | 资源开销大，训练复杂          |
| DPO      |      否 |          不需要显式 RM |         不需要 | 简单稳定，训练便宜                  | 不能在线探索              |
| GRPO     |      是 |       可用规则 reward |         不需要 | 比 PPO 省显存，适合 RLVR          | group 全 0 / 全 1 无梯度 |
| DAPO     |      是 |      多用于规则 reward |         不需要 | 解决长 CoT RL 中的无效样本、熵塌陷、长度问题 | 系统复杂，采样开销更高         |

---

## 10. 强化学习资源占用

### 10.1 为什么 LLM RL 很贵

LLM RL 贵在三个方面：

```text
1. rollout generation 很慢
2. 多模型同时存在，占显存
3. 长序列反向传播占显存
```

训练时 attention 复杂度近似：

$$
O(BL^2d)
$$

KV cache 近似：

$$
O(B \cdot L \cdot n_{\text{layers}} \cdot d_{\text{hidden}})
$$

其中：

| 符号  | 含义              |
| --- | --------------- |
| $B$ | batch size      |
| $L$ | sequence length |
| $d$ | hidden size     |

---

### 10.2 不同算法的资源占用

#### SFT

```text
policy model forward + backward
```

最便宜。

---

#### DPO

```text
policy chosen / rejected forward + backward
reference chosen / rejected forward
```

比 SFT 贵，因为每个样本有 chosen / rejected 两个回答，还要算 reference logprob。

但 DPO 不需要在线 rollout，所以比 PPO / GRPO 便宜很多。

---

#### PPO-RLHF

```text
actor rollout
reward model scoring
critic forward / backward
actor forward / backward
reference model KL
```

通常最贵。

因为需要：

* actor；
* critic；
* reward model；
* reference model；
* old policy；
* optimizer state；
* rollout 缓存。

---

#### GRPO

```text
actor rollout G times
reward scoring
actor update
reference KL optional
```

省掉 critic，但每个 prompt 要生成 $G$ 个回答。

rollout token 数大约是：

$$
N_{\text{tokens}}
=================

B \times G \times T
$$

其中 $T$ 是平均生成长度。

所以：

> GRPO 省 critic 显存，但增加 rollout 成本。

---

### 10.3 VLM RL 比 LLM RL 更贵的原因

VLM 中总 token 长度是：

$$
L = L_{\text{text}} + L_{\text{vision}}
$$

对于图像：

$$
L_{\text{vision}}
\approx
\frac{H}{p}
\cdot
\frac{W}{p}
$$

对于视频：

$$
L_{\text{vision}}
\approx
N_f
\cdot
\frac{H}{p}
\cdot
\frac{W}{p}
$$

其中：

| 符号    | 含义         |
| ----- | ---------- |
| $N_f$ | 帧数         |
| $H,W$ | 图像高宽       |
| $p$   | patch size |

所以视频 VLM RL 特别贵：

```text
多帧视觉 token + 长文本 CoT + 多次 rollout + 工具调用
```

这也是为什么 video agent RL 一次 rollout 可能非常慢。

---

## 11. LLM RL 和 VLM RL 的区别

| 维度                | LLM RL                    | VLM RL                                 |
| ----------------- | ------------------------- | -------------------------------------- |
| 输入                | 文本 prompt                 | 文本 + 图片 / 视频 / 音频                      |
| 动作                | token / tool call         | token / tool call / 视觉定位 / 帧选择         |
| reward            | 文本答案、数学答案、代码测试            | 答案正确性 + grounding + evidence + OCR/ASR |
| 难点                | 长 CoT、reward hacking、格式控制 | 视觉证据获取、帧采样、跨模态对齐                       |
| 显存                | 主要由文本长度决定                 | 视觉 token + 文本 token 共同决定               |
| rollout           | 生成文本                      | 视觉编码 + 文本生成 + 可能工具调用                   |
| credit assignment | answer-level 到 token      | answer-level 到 token + patch + frame   |
| 典型问题              | 胡编、格式错、过长                 | 不看图、视觉幻觉、证据错位、时间定位错误                   |

---

### 11.1 面试回答模板

> LLM RL 的状态主要是 prompt 和已生成 token，动作是下一个 token；VLM RL 还需要处理视觉输入，因此状态包含图像、视频帧、OCR、ASR、视觉 grounding 等信息。VLM RL 的资源开销更大，因为视觉 token 会显著增加序列长度，视频还会引入多帧冗余。奖励设计也更复杂，不能只看最终答案，还需要考虑视觉证据是否正确、时间片段是否定位准确、工具调用是否有效。因此 VLM RL 更难的地方在于跨模态 credit assignment 和 evidence grounding。

---

## 12. TRL 和 verl 的实现区别

### 12.1 TRL 是什么

TRL 是 Hugging Face 的 post-training 库，支持 SFT、PPO、DPO、GRPO、KTO 等训练器。官方文档中 TRL 提供 PPOTrainer、GRPOTrainer 等接口，并集成 Transformers 生态。([Hugging Face][7])

适合：

```text
单机 / 小规模 / 快速实验 / HuggingFace 生态
```

---

### 12.2 verl 是什么

verl 是面向大规模 LLM post-training 的 RL 框架，是 HybridFlow 的开源实现。官方文档强调它支持复杂 RL dataflow、FSDP、Megatron-LM、vLLM、SGLang、灵活 device mapping 和高吞吐训练。([verl][8])

适合：

```text
多机多卡 / 大规模 GRPO / PPO / DAPO / RLVR
```

---

### 12.3 TRL vs verl 对比

| 维度      | TRL                       | verl                         |
| ------- | ------------------------- | ---------------------------- |
| 生态      | Hugging Face Transformers | ByteDance / HybridFlow / Ray |
| 使用难度    | 简单                        | 更复杂                          |
| 适合规模    | 小到中等规模                    | 中到超大规模                       |
| 算法接口    | Trainer 风格                | RL dataflow 风格               |
| 分布式     | Accelerate / DeepSpeed    | Ray + FSDP / Megatron        |
| rollout | 可接 vLLM，但整体偏易用            | 深度集成 vLLM / SGLang           |
| 资源调度    | 相对简单                      | 灵活 device mapping            |
| 工程目标    | 易用性                       | 高吞吐、大规模、生产级                  |
| 典型使用    | DPO/SFT/小规模 GRPO          | DeepSeek-R1 风格 RLVR、DAPO     |

---

### 12.4 面试回答模板

> TRL 更像 Hugging Face Trainer 的强化学习扩展，优点是简单、生态好、适合快速验证；verl 更像面向大规模 LLM RL 的系统框架，核心是 HybridFlow，把 rollout、reward、advantage、update 等复杂数据流和底层分布式执行解耦。verl 可以集成 FSDP、Megatron、vLLM、SGLang，并用 Ray 做资源调度，更适合多机多卡的大规模 GRPO / DAPO 训练。

---

## 13. verl 底层有哪些效率优化

### 13.1 vLLM 加速 rollout

RL 中 rollout generation 往往是瓶颈。

vLLM 的核心是 **PagedAttention**，把 KV cache 像操作系统分页一样管理，减少碎片和重复拷贝，从而提升吞吐。vLLM 论文指出，PagedAttention 让 vLLM 在相同延迟水平下比 FasterTransformer / Orca 等系统有 2-4 倍吞吐提升。([arXiv][9])

verl 支持 vLLM / TGI 等 rollout backend，官方 performance tuning 文档也把 rollout generation throughput 作为首要优化项。([verl][10])

---

### 13.2 Ray 做资源调度

verl 通过 Ray 管理不同 worker：

```text
actor worker
rollout worker
reference worker
reward worker
critic worker
```

Ray 的作用：

* 多机资源管理；
* GPU placement；
* actor / rollout / reward 并行；
* 异步任务调度；
* 降低工程复杂度。

---

### 13.3 FSDP / Megatron 并行训练

verl 可以接入：

```text
PyTorch FSDP
Megatron-LM
Tensor Parallel
Pipeline Parallel
Sequence Parallel
```

作用：

* 切分模型参数；
* 切分 optimizer state；
* 切分梯度；
* 支持大模型训练；
* 降低单卡显存压力。

---

### 13.4 3D-HybridEngine / resharding

verl 文档提到，3D-HybridEngine 可以减少 training 和 generation 阶段切换时的内存冗余与通信开销。([verl][8])

为什么需要 resharding？

```text
训练时：模型适合 FSDP / Megatron 切分
推理时：模型适合 vLLM tensor parallel + KV cache
```

两者最佳并行方式不同，所以要在训练和 rollout 之间高效转换参数布局。

---

### 13.5 sequence packing / remove padding

LLM RL 中不同回答长度差异很大。

如果直接 padding：

```text
短样本也要补到最长样本长度
```

浪费大量算力。

verl performance guide 建议使用：

```text
use_remove_padding=True
use_dynamic_bsz=True
```

用于减少 padding、动态调整 batch，提高吞吐。([verl][10])

---

### 13.6 异步 rollout / train

同步 RL 的问题：

```text
一个超长 response 会拖慢整个 batch
```

异步思路：

```text
rollout worker 持续生成
train worker 持续训练
两者用 buffer 解耦
```

好处：

* 减少等待长尾样本；
* 提高 GPU 利用率；
* 更适合长 CoT。

---

## 14. vLLM 推理框架补充

注意这里是小写 **vLLM**，不是视觉语言模型 VLM。

### 14.1 vLLM 解决什么问题

LLM serving 的瓶颈：

```text
KV cache 很大
请求长度动态变化
显存碎片严重
静态 batching GPU 利用率低
```

vLLM 的核心：

```text
PagedAttention + continuous batching
```

PagedAttention：

$$
KV\ Cache \rightarrow \text{paged blocks}
$$

即 KV cache 不再要求连续显存，而是分页管理。

---

### 14.2 vLLM 在 RL 中为什么重要

在 GRPO / DAPO 中：

$$
\text{rollout tokens} = B \times G \times T
$$

如果 $G=16$，$T=8192$，rollout 会非常慢。

所以 verl 通常用 vLLM 做高速生成：

```text
policy weights
    ↓ sync
vLLM rollout engine
    ↓ generate
responses
    ↓ reward / advantage
training engine
```

---

## 15. VLM 发展历程：CLIP → BLIP → LLaVA → Qwen-VL → DeepSeek-VL

### 15.1 CLIP

CLIP 是 2021 年 OpenAI 提出的对比式图文预训练模型，用 4 亿图文对学习图像和文本的共享语义空间，支持 zero-shot 分类。([arXiv][11])

核心结构：

```text
image encoder
text encoder
contrastive learning
```

对比学习目标：

$$
s_{ij}
======

\frac{
f_I(I_i)^\top f_T(T_j)
}{
\tau
}
$$

图像到文本 loss：

$$
\mathcal{L}_{I2T}
=================

-\frac{1}{N}
\sum_i
\log
\frac{
\exp(s_{ii})
}{
\sum_j \exp(s_{ij})
}
$$

文本到图像 loss：

$$
\mathcal{L}_{T2I}
=================

-\frac{1}{N}
\sum_i
\log
\frac{
\exp(s_{ii})
}{
\sum_j \exp(s_{ji})
}
$$

总 loss：

$$
\mathcal{L}
===========

\frac{1}{2}
(\mathcal{L}*{I2T}+\mathcal{L}*{T2I})
$$

CLIP 创新点：

* 用自然语言监督视觉表征；
* 图文双塔结构；
* zero-shot 分类；
* 奠定后续 VLM 的视觉 encoder 基础。

---

### 15.2 BLIP

BLIP 解决的问题：

```text
早期 VLP 模型往往只擅长理解或生成，难以统一。
```

BLIP 提出 unified vision-language understanding and generation，并用 captioner + filter 清洗 noisy web image-text pairs。([arXiv][12])

结构：

```text
image encoder
text encoder
text decoder
captioner
filter
```

创新点：

* 同时支持理解任务和生成任务；
* bootstrapping caption；
* 过滤噪声图文对；
* 可迁移到 video-language zero-shot 场景。

---

### 15.3 BLIP-2

BLIP-2 的核心是：

```text
Frozen image encoder
+
Q-Former
+
Frozen LLM
```

BLIP-2 认为端到端训练大规模 VLM 成本过高，因此冻结视觉编码器和 LLM，只训练轻量 Q-Former 来桥接视觉和语言。([arXiv][13])

创新点：

* 冻结大模型，降低训练成本；
* Q-Former 作为视觉查询模块；
* 两阶段预训练：

  1. vision-language representation learning；
  2. vision-to-language generative learning；
* 让 VLM 进入“视觉 encoder + adapter + LLM”的范式。

---

### 15.4 LLaVA

LLaVA 是 Visual Instruction Tuning 的代表工作。它使用 GPT-4 生成多模态 instruction data，并将视觉 encoder 接到 LLM 上做视觉指令微调。([arXiv][14])

典型结构：

```text
CLIP ViT
    ↓
MLP projector
    ↓
Vicuna / LLaMA
```

创新点：

* 首次系统探索 multimodal instruction tuning；
* 用语言模型生成视觉指令数据；
* 让 VLM 具备对话式图像理解能力；
* 开源生态强，成为大量 VLM baseline。

---

### 15.5 Qwen-VL

Qwen-VL 基于 Qwen-LM 加入视觉能力，设计了 visual receptor、input-output interface、三阶段训练流程和多语言多模态语料，并强化 grounding 和 text reading 能力。([arXiv][15])

创新点：

* Qwen 语言底座；
* visual receptor；
* 图文统一输入输出接口；
* 支持 OCR、grounding、视觉问答；
* 中文和多语言能力较强。

---

### 15.6 Qwen2-VL

Qwen2-VL 的核心升级是 **Naive Dynamic Resolution**，让模型可以把不同分辨率图片动态转换成不同数量的 visual tokens，而不是固定分辨率处理。([arXiv][16])

创新点：

* 动态分辨率；
* 更适合真实世界不同尺寸图片；
* 更强视频理解；
* 更好的视觉 token 效率；
* 继续强化 OCR、文档、图表理解。

---

### 15.7 Qwen2.5-VL

Qwen2.5-VL 强调：

```text
visual recognition
object localization
document parsing
long-video comprehension
visual agent
computer use / phone use
```

Qwen2.5-VL 技术报告称其在视觉识别、精确定位、文档解析和长视频理解方面有明显提升。([arXiv][17]) Qwen 官方博客还强调 Qwen2.5-VL 可作为 visual agent，具备 computer use 和 phone use 能力。([Qwen Studio][18])

创新点：

* 更强 grounding；
* 更强文档理解；
* 更强长视频理解；
* 从 VQA 走向 visual agent；
* 可进行 UI 操作、手机/电脑使用。

---

### 15.8 Qwen3-VL

Qwen3-VL 技术报告显示，其支持 256K interleaved multimodal context，模型家族包含 dense 和 MoE 版本，并引入 interleaved-MRoPE、DeepStack、多模态时间对齐等升级。([arXiv][19])

创新点：

* 256K 多模态长上下文；
* dense + MoE 双路线；
* interleaved-MRoPE；
* DeepStack 多层视觉特征融合；
* 更强视频时间定位；
* 更强 agentic decision-making。

---

## 16. DeepSeek 系列重点

### 16.1 DeepSeek-VL

DeepSeek-VL 面向真实世界视觉语言理解，覆盖 web screenshots、PDF、OCR、charts、knowledge-based content 等场景，并采用 hybrid vision encoder 处理高分辨率图像。([arXiv][20])

创新点：

* 强调真实世界数据；
* 覆盖截图、PDF、OCR、图表；
* hybrid vision encoder；
* 重视保留 LLM 原有语言能力；
* 1.3B / 7B 开源。

---

### 16.2 DeepSeek-VL2

DeepSeek-VL2 是 MoE 视觉语言模型，相比 DeepSeek-VL 有两大升级：

1. 视觉侧：dynamic tiling vision encoding；
2. 语言侧：DeepSeekMoE + MLA。

DeepSeek-VL2 使用动态 tiling 处理高分辨率和不同长宽比图像，并用 Multi-head Latent Attention 压缩 KV cache，提高推理效率。([arXiv][21])

创新点：

* MoE VLM；
* dynamic tiling；
* MLA 降低 KV cache；
* OCR、文档、表格、图表、grounding 能力增强；
* activated parameters 较少但性能强。

---

### 16.3 DeepSeek-V3

DeepSeek-V3 是 LLM，不是 VLM，但它是 DeepSeek 后续推理模型的重要底座。

DeepSeek-V3 是 671B 总参数、每 token 激活 37B 的 MoE 模型，采用 MLA 和 DeepSeekMoE，并提出 auxiliary-loss-free load balancing 与 multi-token prediction objective。([arXiv][22])

创新点：

* 大规模 MoE；
* MLA；
* DeepSeekMoE；
* auxiliary-loss-free balancing；
* multi-token prediction；
* 高效训练和推理。

---

### 16.4 DeepSeek-R1

DeepSeek-R1 是 reasoning RL 的代表。

核心路线：

```text
DeepSeek-R1-Zero:
    base model 直接大规模 RL
    ↓
    自发出现长 CoT、反思、自验证
    但有可读性差、语言混杂问题

DeepSeek-R1:
    cold-start SFT
    ↓
    RL
    ↓
    rejection sampling / SFT
    ↓
    再 RL
```

DeepSeek-R1 论文称，R1-Zero 不经过 SFT 直接通过大规模 RL 获得强推理能力，而 R1 加入 cold-start data 和多阶段训练来改善可读性与推理表现。([arXiv][5])

创新点：

* RL 激发 reasoning；
* rule-based reward；
* GRPO；
* self-verification；
* reflection；
* cold-start + multi-stage training；
* 蒸馏出多个小模型。

---

## 17. VLM 发展主线总结

### 17.1 一条线背下来

```text
CLIP:
    图文对比学习，解决视觉 zero-shot 表征

BLIP:
    统一理解和生成，用 captioner/filter 清洗数据

BLIP-2:
    冻结视觉 encoder 和 LLM，用 Q-Former 桥接模态

LLaVA:
    visual instruction tuning，让 VLM 具备图像对话能力

Qwen-VL:
    强化 OCR、grounding、多语言和通用视觉问答

Qwen2-VL / Qwen2.5-VL:
    动态分辨率、文档解析、长视频、visual agent

Qwen3-VL:
    长上下文、多模态 MoE、视频时序对齐、agentic reasoning

DeepSeek-VL / VL2:
    真实世界视觉理解、高分辨率、dynamic tiling、MoE、MLA

DeepSeek-R1:
    虽不是 VLM，但代表 reasoning RL 范式，用 GRPO/RLVR 激发长 CoT
```

---

## 18. 高频追问合集

### 18.1 为什么 PPO 在 LLM RL 中不稳定？

答：

> 因为 LLM 动作空间是整个词表，序列很长，reward 通常是稀疏的 final reward，policy update 很容易导致分布漂移。PPO 虽然用 clip 和 KL 控制更新幅度，但仍然需要同时训练 actor 和 critic，critic 在长文本生成任务中估值困难，reward model 也可能被 hack，所以训练不稳定且资源开销大。

---

### 18.2 为什么 DPO 比 PPO 简单？

答：

> DPO 不需要在线 rollout，不需要训练 reward model，也不需要 critic。它直接利用 chosen / rejected 偏好对，通过提高 chosen 相对 reference model 的概率、降低 rejected 相对 reference model 的概率来完成偏好优化。因此 DPO 实现上接近监督学习，比 PPO-RLHF 简单稳定。

---

### 18.3 为什么 GRPO 适合数学推理？

答：

> 数学推理通常可以用 rule-based reward 判断最终答案对错，不一定需要训练 reward model。GRPO 对同一个题目采样多个答案，通过组内相对 reward 计算 advantage，不需要 critic，显存更低。同时多样化采样可以让模型探索不同推理路径，因此适合 RLVR 场景。

---

### 18.4 GRPO 的缺点是什么？

答：

> 第一，rollout 成本高，因为每个 prompt 要采样多个回答。第二，如果 group 内全对或全错，advantage 为 0，没有有效梯度。第三，只用 final reward 时 credit assignment 很粗，模型不知道哪一步推理导致对错。第四，长 CoT 场景中还会有长度偏置、熵塌陷和 overlong reward noise，需要 DAPO 等方法改进。

---

### 18.5 DAPO 解决了 GRPO 哪些问题？

答：

> DAPO 针对长 CoT GRPO 提出四个技巧。Clip-Higher 提高 upper clip，缓解熵塌陷，增强探索；Dynamic Sampling 过滤全 0 或全 1 的无效 group，保证 batch 中有有效梯度；Token-Level Policy Gradient Loss 解决长回答 token 梯度被稀释的问题；Overlong Reward Shaping 避免简单惩罚截断样本带来的 reward noise。

---

### 18.6 verl 为什么比 TRL 更适合大规模 RL？

答：

> TRL 更偏 Trainer API，适合快速实验；verl 面向大规模 RL post-training，核心是 HybridFlow，把算法数据流和底层分布式执行解耦。它可以用 Ray 做资源调度，用 FSDP/Megatron 做训练并行，用 vLLM/SGLang 做 rollout 加速，还支持灵活 device mapping、resharding、remove padding、dynamic batch size 等优化，所以更适合多机多卡 GRPO/DAPO。

[1]: https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with ..."
[2]: https://arxiv.org/abs/2305.18290 "[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[3]: https://arxiv.org/abs/2402.03300 "[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
[4]: https://arxiv.org/html/2503.14476v1 "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
[5]: https://arxiv.org/abs/2501.12948?utm_source=chatgpt.com "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
[6]: https://huggingface.co/docs/trl/en/grpo_trainer "GRPO Trainer · Hugging Face"
[7]: https://huggingface.co/docs/trl/main/en/ppo_trainer "PPO Trainer · Hugging Face"
[8]: https://verl.readthedocs.io/ "Welcome to verl’s documentation! — verl  documentation"
[9]: https://arxiv.org/abs/2309.06180 "[2309.06180] Efficient Memory Management for Large Language Model Serving with PagedAttention"
[10]: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"
[11]: https://arxiv.org/abs/2103.00020?utm_source=chatgpt.com "Learning Transferable Visual Models From Natural Language Supervision"
[12]: https://arxiv.org/abs/2201.12086?utm_source=chatgpt.com "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"
[13]: https://arxiv.org/abs/2301.12597?utm_source=chatgpt.com "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
[14]: https://arxiv.org/abs/2304.08485?utm_source=chatgpt.com "[2304.08485] Visual Instruction Tuning"
[15]: https://arxiv.org/abs/2308.12966?utm_source=chatgpt.com "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"
[16]: https://arxiv.org/html/2409.12191v1?utm_source=chatgpt.com "Qwen2-VL: Enhancing Vision-Language Model's ..."
[17]: https://arxiv.org/abs/2502.13923?utm_source=chatgpt.com "[2502.13923] Qwen2.5-VL Technical Report"
[18]: https://qwen.ai/blog?id=qwen2.5-vl&utm_source=chatgpt.com "Qwen2.5-VL 7B"
[19]: https://arxiv.org/abs/2511.21631?utm_source=chatgpt.com "Qwen3-VL Technical Report"
[20]: https://arxiv.org/abs/2403.05525?utm_source=chatgpt.com "DeepSeek-VL: Towards Real-World Vision-Language Understanding"
[21]: https://arxiv.org/abs/2412.10302?utm_source=chatgpt.com "DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding"
[22]: https://arxiv.org/abs/2412.19437 "[2412.19437] DeepSeek-V3 Technical Report"

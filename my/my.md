# 具身智能学习路线笔记

## 0. 我们当前的位置

我们现在不是从零泛读具身智能，而是已经有一条比较清楚的工程主线：

```text
XR 遥操采集数据
-> LeRobot 数据格式 / 训练 / 评估 / 部署
-> ACT / Diffusion Policy / pi0 / pi0.5 等低层策略
-> subtask + image + state 的短程执行
-> 缺少上层规划、上下文、记忆、失败恢复
-> HIL-SERL 真机强化，把困难任务继续练熟
-> 后续关注 World Model / World Action Model / MEM
```

所以论文学习不应该按“具身智能大全”来铺开，而应该围绕这条链路解决问题：

- 数据怎么采：XR 遥操数据如何变成可训练、可复用、可回放的数据集。
- 技能怎么学：ACT / DP / VLA 如何从 demonstration 学到短程操作。
- 任务怎么拆：上层 planner 如何把自然语言任务拆成 subtask。
- 失败怎么处理：success detector、failure recovery、human intervention 如何进入闭环。
- 能力怎么提升：HIL-SERL / RL from experience 如何在真机上补强。
- 长任务怎么做：context / memory / world model 如何支持多步任务和预测动作后果。

## 1. 核心概念地图

### VLA

VLA 指 Vision-Language-Action，典型形式是：

```text
image / video + language + robot state -> action
```

需要注意的是，ACT 和 Diffusion Policy 严格来说更像 visuomotor imitation policy，不一定天然包含 language。它们如果加上 task language 或 subtask language 条件，才更接近 VLA。

对我们来说可以这样分：

- ACT：短程动作 chunk，适合稳定复现遥操轨迹。
- Diffusion Policy：生成连续动作轨迹，适合接触丰富、多峰动作分布的任务。
- pi0 / pi0.5 / OpenVLA / RT-2：更典型的 VLA，重点是语言条件、多任务、跨场景泛化。
- SmolVLA：工程上更轻量，适合和 LeRobot 生态结合。

### 上层规划

当前 LeRobot policy 多数更像：

```text
subtask + image + state -> action
```

但系统还需要回答：

- 用户一句话任务怎么拆成多个 subtask？
- 当前 subtask 是否已经成功？
- 失败后是重试、换策略、还是请求人类接管？
- 多个物体、多步任务的状态如何记录？

这层可以先做成工程上可控的 hierarchical system：

```text
task instruction
-> VLM/LLM planner 生成 subtask
-> subtask + image + state 输入低层 policy
-> 执行动作 chunk
-> success/failure detector
-> next subtask / recovery / human intervention
```

### RL / HIL-SERL

HIL-SERL 是我们近期最贴近真机的 RL 路线。它不是从零开始让机器人乱试，而是：

```text
少量 demonstration
-> 训练初始策略 / 奖励分类器
-> SAC actor-learner 真机探索
-> 人类随时接管纠正
-> 把 intervention 和成功经验重新用于训练
```

建议优先吃透 HIL-SERL 和 SAC，再补 PPO / sim-to-real。原因是我们已经有 XR 遥操和真机条件，HIL-SERL 比纯仿真 PPO 更直接服务当前系统。

### World Model

World Model 关注“动作会让世界怎么变化”：

```text
current observation + action -> next state / future image / latent future
```

它可以用于：

- 训练时做辅助预测任务。
- 推理时模拟多个候选动作。
- 生成 synthetic robot data。
- 做失败预测和风险评估。

对我们来说，World Model 不一定马上工程落地，但它能指导我们理解：为什么只靠 reactive policy 很难处理长任务、遮挡、失败恢复和动作后果预测。

### WA / WAM

WA / WAM 可以统一放到 World Action Model 这一类。它比普通 World Model 更进一步：不仅预测世界，还把预测和动作决策结合起来。

可以简单理解为：

```text
VLA：看到当前，直接输出动作
WM：给定动作，预测未来
WAM：结合未来预测来选择或生成动作
```

如果未来我们要做上层规划增强，WAM 可能对应这些功能：

- 生成 subgoal image。
- 预测执行一个 subtask 后的画面。
- 为多个候选 action chunk 打 value。
- 在真实执行前做 imagined rollout。

### MEM / Context / Memory

MEM 和 context 解决的是长任务中“当前帧不够用”的问题。

机器人需要记住：

- 已经完成了哪些 subtask。
- 哪个抽屉打开过。
- 哪个物体之前被移动到了哪里。
- 哪些动作失败过。
- 人类在哪些地方接管过。
- 当前任务的约束和偏好是什么。

这对我们很关键，因为现在的 policy 输入一般是短上下文：`subtask + image + state`。后续如果做复杂任务，需要补：

- episode-level memory
- subtask history
- failure history
- human intervention history
- goal / subgoal image
- object-level state tracking

## 2. 推荐学习顺序

### 工程优先路线

这条最适合我们当前项目：

```text
1. LeRobot 数据格式、训练、评估、部署
2. ACT 原理和 LeRobot 实现
3. Diffusion Policy 原理和 LeRobot 实现
4. pi0 / pi0.5 / SmolVLA
5. 上层 planner：task -> subtask -> policy
6. HIL-SERL：真机强化和人类接管闭环
7. MEM / context：长任务状态管理
8. World Model / WAM：预测动作后果和规划增强
```

### 论文优先路线

如果要按论文建立理论体系：

```text
1. VLA survey
2. ACT / ALOHA
3. Diffusion Policy
4. RT-1 / RT-2
5. OpenVLA / Octo / DROID
6. pi0 / pi0.5 / pi*0.6
7. HIL-SERL
8. World Model survey
9. World Action Model survey
10. MEM / context 相关论文
```

### 短期最该补的能力

短期优先级不是追最大模型，而是把系统闭环补齐：

- 统一 XR -> LeRobot dataset 的字段。
- 明确每个 episode 的 task / subtask / success / failure。
- 为 ACT 和 DP 建立稳定 baseline。
- 增加 success detector。
- 增加 simple planner 或 state machine。
- 把 human intervention 记录进数据。
- 选择 1 到 2 个任务跑 HIL-SERL 闭环。

## 3. 综述入口

### World Action Models

[arxiv.org/pdf/2605.12090](https://arxiv.org/pdf/2605.12090) World Action Models: The Next Frontier in Embodied AI

这篇可以作为理解 WA / WAM 的总入口。传统 VLA 更偏向“看见当前图像和状态，然后直接输出动作”，而 World Action Model 关注模型能不能同时理解世界变化、预测动作后果，并基于这种预测来选择动作。

对我们现在的系统来说，它最有价值的地方是补上 ACT / DP / pi0 / pi0.5 上面缺的那一层：长任务里需要判断下一步做什么、失败后怎么恢复、执行动作会不会把物体推歪。WAM 可以帮助我们思考是否引入 future image、subgoal image、value prediction 或 imagined rollout。

阅读时重点看：

- 它怎么表示 action：动作 token、连续控制、轨迹、latent action，还是 video latent。
- 它怎么建模未来：预测下一帧、未来状态、子目标图像，还是价值函数。
- 它怎么用于控制：直接输出动作，还是生成多个候选动作后做 planning。

### VLA Survey

[arxiv.org/abs/2405.14093](https://arxiv.org/abs/2405.14093) A Survey on Vision-Language-Action Models for Embodied AI

这篇是 VLA 方向最适合作为第一入口的综述之一。它的价值是把 VLA 的基本概念、模型结构、训练方法、任务类型和评测方式统一起来。对我们来说，它适合回答“ACT / DP / pi0 / OpenVLA / RT-2 到底该放在同一张图里的哪个位置”。

重点看：

- VLA 和 VLM、传统 visuomotor policy 的边界。
- action 表示方式：离散 token、连续动作、action chunk、diffusion / flow matching。
- high-level planning 和 low-level control 是否解耦。
- manipulation、navigation、mobile manipulation 的任务差异。

### Manipulation VLA Survey

[arxiv.org/abs/2508.13073](https://arxiv.org/abs/2508.13073) Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey

这篇更聚焦“基于大 VLM 的机械臂操作 VLA”，比泛 Embodied AI 综述更贴近我们现在的机械臂系统。它把 VLA 分成 monolithic 和 hierarchical 两类：前者是一个大模型直接从多模态输入到动作，后者是显式拆成规划、目标、中间表示、执行策略。

这篇对我们尤其有用，因为我们当前系统天然就是 hierarchical：

```text
上层：subtask / planner / context / success detector
下层：ACT / DP / pi0 / pi0.5 policy
```

### World Model Survey

[ntumars.github.io/wm-robot-survey](https://ntumars.github.io/wm-robot-survey/) World Model for Robot Learning: A Comprehensive Survey

这篇是 World Model 方向很好的总入口。它不是只讲 VLA，而是从 robot learning 角度讲 world model 怎么服务于 policy learning、planning、simulation、evaluation、data generation 和 robotic video generation。

对我们来说，它适合回答：

- world model 到底是预测图像、状态、latent dynamics，还是 learned simulator。
- world model 是训练时用、推理时用，还是作为数据生成器用。
- 什么情况下 world model 值得上，什么情况下 ACT/DP 更划算。

### Real-World VLA Review

[doi.org/10.1109/ACCESS.2025.3609980](https://doi.org/10.1109/ACCESS.2025.3609980) Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications

这篇适合从真实机器人落地角度看 VLA。它比较关注 VLA 设计策略、训练策略、实现方式，以及从 CNN、CLIPort、RT-1/RT-2、OpenVLA 到 pi0 / pi0.5 / GR00T N1 这些路线的演进。

对我们来说，这篇适合做路线对照：哪些模型只是论文 demo，哪些开始接近真实机器人部署；哪些适合短任务，哪些适合长任务；哪些需要很大数据，哪些能在自采数据上 fine-tune。

### Foundation Model + Robot Learning

[www.sciencedirect.com/science/article/abs/pii/S0925231225006356](https://www.sciencedirect.com/science/article/abs/pii/S0925231225006356) Robot learning in the era of foundation models: a survey

这篇范围更大，适合补“foundation model + robot learning”的背景。它会把 LLM/VLM/VLA、数据集、仿真器、robot learning 框架放在一起讲。它不一定最贴近我们的工程，但适合作为背景补全，尤其是给团队成员建立共同语言。

### World Models for Manipulation

[onlinelibrary.wiley.com/doi/10.1002/smb2.70053](https://onlinelibrary.wiley.com/doi/10.1002/smb2.70053) World Models for Robotic Manipulation: A Survey

这篇更聚焦 manipulation 里的 world model。它适合我们判断一个现实问题：对于桌面机械臂任务，究竟什么时候应该上 world model，什么时候继续用 ACT/DP/HIL-SERL 更直接。

## 4. 基于问题的专栏

[sinwang20.github.io/blog/robot-icl-zh](https://sinwang20.github.io/blog/robot-icl-zh/#%E4%B8%80%E4%B8%BA%E4%BB%80%E4%B9%88%E7%AD%96%E7%95%A5%E9%9C%80%E8%A6%81%E4%B8%8A%E4%B8%8B%E6%96%87)

这篇适合放在 MEM / context 主题下读。它讨论的是为什么机器人策略需要上下文，以及上下文如何影响策略泛化。对我们来说，它能直接对应一个现实问题：如果 policy 只看到当前图像和当前 state，它很难知道之前做过什么、失败过什么、当前任务进度是什么。

建议结合我们自己的数据格式一起看：

- episode 是否记录完整任务上下文。
- subtask 是否有明确 id 和自然语言描述。
- 是否记录上一次动作、上一次失败、人工接管点。
- 是否能把历史压缩成 policy 可用的 context。

## 5. 主流系列和团队

### Physical Intelligence / pi

[www.pi.website](https://www.pi.website/)

[www.pi.website/blog/pistar06](https://www.pi.website/blog/pistar06) **pi*0.6: a VLA that Learns from Experience**

Physical Intelligence 的 pi 系列是目前最值得跟的一条 VLA 主线，原因是它一直围绕真实机器人、真实任务、多任务泛化和数据闭环在推进。pi0 / pi0.5 更接近“通过大规模机器人数据训练通用 VLA”，pi*0.6 则开始强调从机器人自主经验中继续学习，把 imitation learning 和 RL 接起来。

这条线和我们最贴的地方在于：我们现在有 XR 遥操数据，可以先走 imitation learning，把 ACT / DP / pi0 / pi0.5 跑通；然后对失败率高、接触复杂、需要精细调整的任务，再用类似 pi*0.6 的思路引入真机经验、奖励反馈和人类纠正。

建议学习顺序：

- pi0：理解 VLA 的基本输入输出形式，重点看 image/language/state 到 action 的映射。
- pi0.5：看跨环境、跨任务、泛化和数据组织方式。
- pi*0.6：重点看 experience learning，尤其是离线 RL、任务微调、机器人自主数据、人类纠错和 reward 信号怎么合起来。

对工程实现的启发：

- XR 采集的数据不仅要服务于 BC/ACT/DP，也要保留失败、纠正、恢复这些片段。
- 数据格式中最好显式记录 subtask、success/failure、human intervention、reward 或 preference。
- 后续如果做 HIL-SERL，可以把它看成 pi*0.6 方向的轻量真机版本。

### Hugging Face LeRobot

LeRobot 应该是我们的工程主干。它的价值不是提出一个最前沿概念，而是把 dataset、policy、training、evaluation、teleoperation、real robot deployment 串成可复用工具链。对我们来说，XR 遥操采数、ACT/DP/pi0/pi0.5 训练、策略回放、真机部署都应该尽量贴着 LeRobot 的数据结构和接口走。

建议学习重点：

- dataset schema：episode、observation、state、action、task、metadata。
- ACT：作为 action chunking baseline。
- Diffusion Policy：作为连续轨迹和接触任务 baseline。
- pi0 / pi0.5 / SmolVLA：作为语言条件和多任务泛化方向。
- HIL-SERL：作为真机强化和人类纠错闭环。

如果后面要引入 MEM/context，LeRobot 数据里就要提前保留 task history、subtask id、失败标签、人工接管片段、恢复动作等字段。

### Google DeepMind

[deepmind.google/models/gemini-robotics](https://deepmind.google/models/gemini-robotics/)

RT-1、RT-2、Gemini Robotics / Gemini Robotics-ER 重点学 VLA + 高层 embodied reasoning。Google DeepMind 这条线最适合补“上层规划和具身推理”：任务分解、空间理解、场景理解、失败解释和工具调用。

对我们的系统设计启发：

- 上层 planner 不一定直接输出连续动作，更适合输出 subtask、目标对象、目标区域、约束和成功条件。
- 低层 ACT/DP/pi policy 只负责短程可执行技能。
- VLM/MLLM 可以做 scene parsing、subtask generation、failure explanation 和 recovery proposal。

### Stanford / UC Berkeley

OpenVLA、Octo、DROID、BridgeData，重点学开源 VLA 和数据规模化。

这条线是开源机器人学习生态的主线，尤其适合我们这种希望自己采数据、自己训练、自己部署的团队。OpenVLA、Octo、DROID、BridgeData 之间可以连起来看：数据集怎么组织，模型怎么预训练，如何 fine-tune 到新机器人，如何评价泛化。

重点理解：

- DROID / BridgeData：大规模真实机器人数据应该怎么采、怎么标注、怎么跨任务复用。
- Octo：通用机器人 policy 如何用多机器人数据训练，并适配新任务。
- OpenVLA：开源 VLA 如何把视觉语言模型和动作输出接起来。

### NVIDIA

[research.nvidia.com/labs/cosmos-lab/cosmos-policy](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/) Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

NVIDIA 的 Cosmos Policy 很适合放在 WA/WAM/World Model 这条线里学。它的核心思想是：不要只把视频模型当作视觉特征提取器，而是把预训练视频模型通过 post-training 改造成能生成动作、预测未来状态、估计 value 的机器人策略。

这和 ACT / DP 最大的区别是：ACT / DP 主要学习 demonstration 里的动作分布；Cosmos Policy 还试图利用视频模型已有的时间、运动和物理先验，让模型具备“预测执行后会发生什么”的能力。

学习重点：

- latent frame injection 如何把 action/state/value 编进视频模型 latent。
- future image prediction 和 action chunk 如何一起生成。
- direct policy 和 model-based planning 两种部署方式的区别。

### Tencent Tairos

[tairos.tencent.com/openSourceModels](https://tairos.tencent.com/openSourceModels)

Tencent Tairos 可以作为国内开源 VLA / 机器人基础模型路线的观察对象。它的价值在于看国内团队如何定义模型接口、数据格式、benchmark、部署方式，以及是否能和 Hugging Face / LeRobot 这样的工程生态接上。

阅读或调研时建议重点看：

- 模型是否开源权重，还是只开放 demo/API。
- 支持哪些机器人形态：机械臂、双臂、移动操作、灵巧手。
- 输入输出格式是否能映射到我们已有的 LeRobot dataset。
- 是否支持语言条件、状态输入、动作 chunk 或连续控制。
- 有没有真实机器人实验，而不是只在仿真或视频里展示。

### LingBot / Robbyant

[www.robbyant.com](https://www.robbyant.com/)

LingBot / Robbyant 这一类更偏国内具身智能产品化路线，重点不是单篇论文，而是观察它们如何把语言交互、移动操作、机械臂执行和场景任务包装成系统。它适合作为“产品形态”和“系统架构”的参考，而不是最底层 policy 的唯一技术来源。

可以重点关注：

- 任务是不是从自然语言开始，而不是从固定 subtask 开始。
- 是否有上层任务规划、技能库、状态机、记忆模块。
- 机器人失败时怎么恢复，是重新询问人、重新规划，还是直接重试。
- 它的数据闭环怎么做：遥操、用户反馈、日志回放、在线修正。

### Generalist

[generalistai.com](https://generalistai.com/)

Generalist 这一类可以放在“机器人基础模型创业公司/通用智能体路线”里观察。它不一定是我们近期工程主线，但适合用来判断行业趋势：通用机器人模型到底在强调数据规模、模型架构、仿真、遥操硬件，还是人类反馈闭环。

调研时建议看：

- 它是否强调 generalist robot policy，还是只做特定场景 demo。
- 是否公布数据来源、机器人平台、任务覆盖范围。
- 有没有和 VLA、world model、memory/context、RL from experience 相关的技术细节。
- 是否能给我们提供系统设计上的启发，比如技能库、云端训练、远程遥操、数据闭环。

## 6. 持续追踪网站

[github.com/Jiaaqiliu/Awesome-VLA-Robotics](https://github.com/Jiaaqiliu/Awesome-VLA-Robotics)

VLA 论文、模型、数据集、benchmark 的综合清单，适合每周扫一次。它的价值是广，能快速看到有哪些新模型、新数据集、新 benchmark 出现。

[github.com/jonyzhang2023/awesome-embodied-vla-va-vln](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)

这个列表覆盖 VLA、WAM、VLN、VA、MLLM-based embodied learning，分类比较接近我们现在关心的概念栈。里面已经把 WAM、MEM、VLA-RL、sim-to-real、benchmark 等方向放在一起，适合用来追 2026 新东西。

[github.com/wangskyone/awesome-VLA-WAM](https://github.com/wangskyone/awesome-VLA-WAM)

这个更窄，但正好贴近我们下一步：VLA、WAM、agentic robotics、failure detection/correction、efficient VLA。它适合专门追“上层规划、记忆、失败恢复、世界动作模型”。

## 7. 读论文时的统一模板

每看一篇论文，都尽量按同一套问题记录：

```text
1. 这篇解决什么问题？
2. 输入是什么：image / language / state / history / subgoal？
3. 输出是什么：action / action chunk / trajectory / subtask / future image / value？
4. 数据来自哪里：遥操、真实机器人、仿真、视频、混合数据？
5. 学习方式是什么：BC、diffusion、transformer、RL、offline RL、world model？
6. 是否支持 long-horizon task？
7. 是否需要上层 planner？
8. 是否有 success detector 或 failure recovery？
9. 能不能接到 LeRobot？
10. 对我们当前 XR 遥操系统有什么直接启发？
```

## 8. 后续可拆分的技术文档

后面可以继续把这份笔记拆成几份更工程化的文档：

- `01_lerobot_data_schema.md`：XR 遥操数据如何对齐 LeRobot。
- `02_policy_baselines.md`：ACT / DP / pi0 / pi0.5 的适用场景和训练配置。
- `03_planner_and_context.md`：上层 subtask planner、memory、success detector。
- `04_hil_serl_loop.md`：真机强化、人工接管、奖励分类器和数据回流。
- `05_world_model_wam.md`：World Model / WAM / MEM 的前沿路线和可落地点。

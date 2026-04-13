# Music Agent Demo

`music-agent-demo` 是一个面向音乐生成的最小闭环 agent demo。它把大模型规划、MiniMax 音乐生成、音频验证和 prompt 迭代串成一个循环，目标不是“一次写出最完美 prompt”，而是通过多轮生成和打分，逐步把 prompt 推向更高质量的结果。

核心思路很直接：

1. 用 LLM 选择一个最合适的生成 skill。
2. 用 LLM 把用户需求编译成结构化 `PromptBrief`。
3. 再用 LLM 把需求编译成一组可验证的 checklist。
4. 调用 MiniMax 生成音频。
5. 用一组 validator skill 对音频逐项打分。
6. 用 verifier agent 总结失败点和保留项。
7. 用这些反馈改写 prompt，进入下一轮。
8. 达到目标分数或达到最大轮数时停止。

## 项目要解决什么问题

音乐生成里，用户需求通常是自然语言，但生成器真正吃得动的是更短、更具体、更偏“可执行”的 prompt。单次 prompt engineering 的问题在于：

- 用户需求往往模糊，比如“更情绪化”“更高级”“更像电影配乐”。
- 生成结果很难只靠主观听感稳定优化。
- 即使某轮优化提升了一个维度，也可能破坏之前已经满足的维度。

这个 demo 的解决方式是：把“听感目标”拆成明确的检查项，让每轮优化都有证据、有约束、有回路。

## 整体架构

主要模块：

- `music_agent_demo/agent.py`
  负责主循环，串起 skill 路由、brief 构建、生成、评估和 prompt refinement。
- `music_agent_demo/evaluator.py`
  负责把需求编译成验证计划、把 checklist 路由给对应 validator，并汇总总分。
- `music_agent_demo/skill_manager.py`
  从 `skills/` 和 `validation_skills/` 目录加载 `SKILL.md`，把 skill 描述和说明暴露给 LLM。
- `music_agent_demo/music_client.py`
  调用 MiniMax `music_generation` 接口，保存请求、响应和音频文件。
- `music_agent_demo/llm_client.py`
  调用 Kimi，支持纯文本完成和 JSON 完成。
- `music_agent_demo/validation_skills/*/tool.py`
  每个 validator 的真实执行逻辑。
- `music_agent_demo/schemas.py`
  定义 `PromptBrief`、`ValidationPlan`、`EvaluationResult`、`AttemptRecord` 等核心数据结构。

可以把主链路看成下面这个循环：

```text
user prompt
  -> skill router
  -> initial brief builder
  -> validation checklist compiler
  -> MiniMax generation
  -> validator skills
  -> verifier summary + next guidance
  -> prompt refinement
  -> next iteration
```

## 运行循环是怎么工作的

入口在 `MusicGenerationAgent.run()`。

每次运行会：

1. 创建一个新的 `runs/<timestamp>-<slug>/` 目录。
2. 选择生成 skill。
3. 构建 `PromptBrief`。
4. 构建 `ValidationPlan`。
5. 如果是 `--dry-run`，到这里结束，只输出规划结果。
6. 否则进入生成-评估循环。

循环内每一轮都做这些事：

1. 用当前 `generation_prompt` 调用 MiniMax 生成音频。
2. 转成 WAV，便于统一验证。
3. 把 checklist 路由到不同 validator。
4. 执行所有 validator，得到逐项结果。
5. 计算加权总分。
6. 让 verifier agent 基于 validator 证据总结：
   - 哪些是硬失败
   - 哪些检查项已经满足，后续要保护
   - 下一轮 prompt 应该怎么改
7. 如果达到 `target_score`，提前停止。
8. 如果还没达到，调用 prompt refiner 生成新的 prompt，进入下一轮。

停止条件有两个：

- `evaluation.total_score >= target_score`
- 已达到 `max_iterations`

默认值来自 `Settings`：

- `max_iterations=3`
- `target_score=0.82`

## Prompt 是如何一步步被优化的

这个项目里，prompt 不是直接由用户输入传给生成器，而是经过三层处理。

### 1. Skill 路由

`route_skill()` 会把所有可用生成 skill 的描述拼成 manifest，再让 Kimi 在这些 skill 中选择一个，或者返回 `NONE`。

它的职责不是“生成 prompt”，而是先决定“这类需求应该按什么思路写 prompt”。

如果 skill 选择错误，后续 prompt 很容易方向偏掉，所以这一步是 prompt 优化的第一层。

### 2. Initial Brief 构建

`build_initial_brief()` 会让 Kimi 直接返回一个结构化 JSON：

- `title`
- `intent_summary`
- `is_instrumental`
- `generation_prompt`
- `lyrics`
- `use_lyrics_optimizer`
- `evaluation_texts`
- `focus_tags`
- `avoid_tags`

这里做了几件关键的 prompt engineering：

- 把模糊需求翻译成更适合 MiniMax 的短 prompt。
- 强制输出 `evaluation_texts`，方便后续做 CLAP 语义对齐。
- 把“是否该让模型自动写歌词”显式化。
- 把“应该强化什么”和“应该避免什么”结构化下来。

如果用户强制 `--instrumental`，代码会覆盖 brief：

- `is_instrumental=True`
- `use_lyrics_optimizer=False`
- `lyrics=""`

也就是说，最终传给生成器的意图并不完全依赖 LLM 输出，还会受到命令行开关的硬约束。

### 3. 失败驱动的 Prompt Refinement

`refine_prompt()` 不会空泛地要求模型“再优化一下”，而是把完整的 attempt history 喂给 LLM。

每轮历史里包括：

- 当前轮使用的 prompt
- 总分
- validator 加权分
- `hard_failures`
- `protected_checks`
- `verifier_summary`
- `next_prompt_guidance`
- 每个 check 的详细结果

refiner prompt 的规则非常关键：

- 保留用户核心意图
- 优先修复失败项，尤其是 hard failures
- 直接使用 verifier guidance 和检查证据
- 保护已经满足的检查项，避免回归
- 删除冗余和冲突描述
- 保持 prompt 简洁、适合生成器

这意味着它不是“自由改写 prompt”，而是一个受检查结果约束的、带回归保护的局部优化器。

## 生成侧 Skills

生成 skill 放在 `music_agent_demo/skills/` 下，只包含 `SKILL.md`，不直接执行代码。它们的作用是给 LLM 一个清晰的“写 prompt 的风格模板”。

当前有 4 个生成 skill：

### `beat_lab`

适合：

- beat
- loop
- hip-hop instrumental
- electronic groove
- club track
- rhythm-first 请求

它强调：

- 节奏、groove、低频、鼓组质感
- producer-oriented 语言
- prompt 要短、直白、偏制作语言

### `genre_fusion`

适合：

- 明确要求融合两种或多种风格
- 跨风格混合需要主动平衡的请求

它强调：

- 用 2 到 3 个核心风格 anchor 控制 prompt
- 说清每个风格分别贡献什么
- 去掉互相冲突的描述

### `instrumental_soundtrack`

适合：

- 配乐
- soundtrack / score
- ambient
- background music
- 游戏、电影、冥想、学习类 instrumental 请求

它强调：

- 默认非 vocal
- 用场景、配器、节奏推进、情绪弧线写 prompt
- 尽量写出 opening / build / climax / landing

### `vocal_songwriter`

适合：

- 有明确 vocal 诉求的完整歌曲
- hook、chorus、唱感、流行歌曲结构

它强调：

- 先写 vocal identity，再写风格和制作
- 没有固定歌词时，不强塞歌词文本
- 把“更情绪化”转成可听见的编排和制作细节

## 评估计划是如何生成的

评估不是写死的。`AudioEvaluator.build_validation_plan()` 会先让 Kimi 把当前需求和 brief 编译成 checklist。

这个 checklist 的特点：

- 每条都是自然语言句子
- 每条都必须足够原子化，方便只由一个 validator 验证
- 一定包含一条“整体语义意图”检查
- 一定包含一条“基本音频健康度”检查
- 其他检查只在需求明确暗示时才添加
- 每条有 `weight`
- 某些检查可以标记为 `hard`

编译完后，代码会把权重归一化，得到 `ValidationPlan.checks`。

## Validator 是怎么工作的

`AudioEvaluator.evaluate()` 做几件事：

1. 先把音频统一转成 WAV。
2. 用 LLM 把 checklist 路由到具体 validator skill。
3. 动态加载对应 `tool.py`，逐项执行 `validate()`。
4. 收集结果并计算加权得分。
5. 再调用 verifier agent 产出适合下一轮 prompt 的指导。

这里的设计点在于：LLM 只负责“编译检查项”和“路由 skill”，具体评分由代码完成。

也就是说，主观判断和客观计算被拆开了：

- LLM 决定“该检查什么”
- 代码决定“怎么测”“怎么打分”

## 当前有哪些 Validator Skills

validator skill 放在 `music_agent_demo/validation_skills/` 下，每个 skill 有 `SKILL.md` 和 `tool.py`。

### `semantic_alignment_validator`

用途：

- 检查生成结果是否在语义上匹配文本意图

实现：

- 使用 CLAP 模型做 audio-text similarity
- 对比的文本包括：
  - 当前 checklist 句子
  - 原始用户请求
  - `brief.intent_summary`
  - `brief.evaluation_texts`
- 最终分数取这些文本相似度的均值
- 默认阈值是 `0.62`

这是整个系统里最核心的“意图对齐”检查器。

### `mix_health_checker`

用途：

- 检查音频是否“明显坏掉”

实现：

- 读取音频后计算：
  - 时长
  - RMS
  - silence ratio
  - clipping ratio
- 各项分数平均成一个健康度分数
- 默认通过阈值是 `0.60`

这是硬护栏类 validator。哪怕风格对了，音频如果太短、太空、太爆，也会被扣下来。

### `tempo_checker`

用途：

- 请求里有明确 BPM 或速度约束时使用

实现：

- 用 `librosa.beat.beat_track()` 估计 tempo
- 从 check 文本里解析目标 BPM
- 默认容差是 `8 BPM`
- 偏差越大，分数越低

### `rhythm_pattern_checker`

用途：

- 请求强调固定 groove 或规律节奏时使用

实现：

- 检测 beat 时间点
- 看 beat interval 的稳定性
- 计算低频 on-beat / off-beat 能量
- 对 `steady_pulse` 或 `four_on_the_floor` 这类模式做评分

### `section_energy_checker`

用途：

- 请求里有“副歌更炸”“后段抬升”“能量弧线”之类结构要求时使用

实现：

- 把整首歌粗分成 4 段
- 用每段 RMS 均值代表段落能量
- 比较后半段峰值和前半段基线
- 看是否存在足够明显的 lift，以及峰值是否出现在后段

### `tone_checker`

用途：

- 请求里有 warm / bright / dark / airy / soft / aggressive 等音色词时使用

实现：

- 计算 spectral centroid、rolloff、低高频能量比
- 再根据不同 tone 类型映射成不同打分逻辑

它不是“语义风格”判定，而是偏频谱层面的音色近似。

### `vocal_presence_checker`

用途：

- 检查是否该有人声
- 检查是否该无人声
- 在简单场景下判断男女声倾向

实现：

- 仍然使用 CLAP 做几个标签之间的对比
- 例如比较：
  - `instrumental music only`
  - `a song with singing vocals`
  - 以及可能的 `female` / `male` vocal 标签
- 看所需标签和对立标签之间的 margin

### `aesthetic_quality_checker`

用途：

- 在启用 Audiobox aesthetics 时做整体制作质量评估

实现：

- 调用 `audiobox_aesthetics` predictor
- 取 `CE / CU / PC / PQ`
- 用固定权重加权求分
- 当前阈值是 `0.55`

注意：代码里在 checklist compiler prompt 中会把 `Audiobox enabled` 传给 LLM，但真正是否可用仍取决于本地依赖和 skill 路由结果。

## 总分是如何计算的

每个 validation check 都有一个权重，代码会先归一化，之后总分按加权平均计算：

```text
validator_score = sum(check.weight * check_result.score)
```

当前实现里：

- `total_score == validator_score`
- verifier 不直接改分，只负责生成“解释”和“下一轮建议”

此外，某些特定 validator 的分数会额外保存到结果结构里，便于分析：

- `clap_mean`
- `clap_scores`
- `aesthetic_score`
- `aesthetic_axes`
- `heuristic_score`
- `heuristics`

## verifier 在质量优化里的作用

verifier 不是另一个打分器，而是“反馈压缩器”。

它读取：

- 原始用户请求
- 当前 generation prompt
- checklist
- 所有 validator 结果

然后输出四类信息：

- `summary`
- `hard_failures`
- `protected_checks`
- `next_prompt_guidance`

这一步很重要，因为 validator 的输出往往是结构化证据，但不一定天然适合喂回 prompt。verifier 的作用就是把“评分结果”翻译成“下一轮可执行的 prompt 修改建议”。

## 数据产物会保存什么

每次 run 都会在 `runs/` 下落盘，便于复盘。

顶层 run 目录通常包含：

- `plan.json`
- `summary.json`
- `attempt_01/`
- `attempt_02/`
- `attempt_03/`

其中：

- `plan.json`
  保存用户请求、skill 选择、brief、validation plan 和是否 dry-run。
- `attempt_xx/request.json`
  发给 MiniMax 的请求体。
- `attempt_xx/response.json`
  MiniMax 返回结果。
- `attempt_xx/generated.mp3`
  生成音频。
- `attempt_xx/generated.wav`
  为评估转码后的音频。
- `attempt_xx/evaluation.json`
  当前轮验证结果。
- `summary.json`
  完整 run 的汇总，包含 best attempt 和所有 attempt。

这个落盘结构让你可以直接回答这些问题：

- 哪一轮 prompt 最好？
- 哪个 validator 一直失败？
- 是生成没跟上，还是评估要求太苛刻？
- 某次 refinement 是否引入了回归？

## PromptBrief 和 payload 的关系

`PromptBrief` 是 agent 内部的中间表示，MiniMax 真正收到的是 `MiniMaxMusicClient.build_payload()` 生成的 payload。

关键映射关系：

- `generation_prompt -> payload.prompt`
- `use_lyrics_optimizer -> payload.audio_setting.lyrics_optimizer`
- `is_instrumental -> payload.audio_setting.is_instrumental`
- `lyrics -> payload.lyrics`

还有一个 MiniMax API 的兼容性细节：即使是纯音乐，代码仍然保证 `lyrics` 字段存在，而且必须是字符串。如果 brief 没有歌词，就回退到 `intent_summary` 或固定占位文本。

## 依赖和运行要求

基础环境定义在 `environment.yml`，核心依赖包括：

- Python 3.10
- `ffmpeg`
- `openai`
- `requests`
- `python-dotenv`
- `numpy`
- `librosa`
- `torch`
- `torchaudio`
- `transformers`

可选依赖：

- `requirements-clap.txt`
  安装 CLAP 相关依赖
- `requirements-audiobox.txt`
  安装 `audiobox_aesthetics`

环境变量至少需要：

- `MINIMAX_API_KEY`
- `MOONSHOT_API_KEY`

常见可调项：

- `MINIMAX_MODEL`
- `KIMI_MODEL`
- `MAX_ITERATIONS`
- `TARGET_SCORE`
- `USE_AUDIOBOX`

## 如何运行

安装环境后，可以直接从项目根目录执行：

```bash
python -m music_agent_demo "一首温暖、克制、适合深夜通勤的华语流行歌"
```

只看规划、不实际生成：

```bash
python -m music_agent_demo "一段带电影感的钢琴氛围配乐" --dry-run
```

指定最大迭代轮数和目标分数：

```bash
python -m music_agent_demo "四踩、稳定推进、带复古合成器的 club track" \
  --iterations 4 \
  --target-score 0.86
```

强制纯音乐：

```bash
python -m music_agent_demo "用于科幻短片结尾的悬浮感配乐" --instrumental
```

## 这个 demo 的边界

它是一个非常清晰的研究/演示型闭环，但还不是完整生产系统。当前边界包括：

- 生成 skill 只有 prompt 指导，没有对应执行代码。
- validator 路由依赖 LLM，稳定性取决于 checklist 质量。
- `total_score` 目前只等于 validator 加权分，还没有把 verifier 或用户反馈纳入最终分数。
- `vocal_presence_checker` 的性别判断是基于 CLAP 标签近似，不是专门的人声分类器。
- `tone_checker` 和 `section_energy_checker` 都是启发式实现，不是高保真音乐学分析。
- 如果本地没装相关依赖，CLAP 或 Audiobox 类 skill 无法运行。

## 如果你要继续扩展这个项目

最自然的扩展方向有四个：

1. 增加新的生成 skill，让路由更细。
2. 增加新的 validator skill，让 checklist 能覆盖更多音乐目标。
3. 把 verifier 的输出进一步结构化，例如显式区分“新增约束”和“保留约束”。
4. 在 `total_score` 中加入更多维度，例如人工评分、偏好模型、用户反馈或多样性指标。

## 代码阅读起点

如果你第一次看代码，建议按这个顺序读：

1. `music_agent_demo/cli.py`
2. `music_agent_demo/agent.py`
3. `music_agent_demo/evaluator.py`
4. `music_agent_demo/schemas.py`
5. `music_agent_demo/music_client.py`
6. `music_agent_demo/validation_skills/*/tool.py`

这样能最快看到“从用户请求到多轮优化”的完整闭环。

# music-agent-demo

一个最小可运行的 music generation agent demo。它借鉴 `GEMS` 的核心思路，但把目标从图像改成了音乐：

`user query -> skill routing -> prompt planning -> music generation -> checklist compilation -> validator routing -> tool-backed skill checks -> verifier agent synthesis -> prompt refinement -> loop`

当前版本追求的是“先跑起来并验证闭环”，而不是完整工程化。为了减少第一次安装失败概率，环境被拆成两层：

- `environment.yml`：核心 agent + CLAP + 音频处理（默认主流程，要求 `torch>=2.6`）
- `requirements-audiobox.txt`：可选安装，负责质量/审美增强评分

## 设计目标

- 和 `GEMS` 一样做循环优化，但只保留单次运行内的历史，不做持久化 memory。
- skill 不是 toolchain，而是 prompt augmentation module。
- 生成端使用 MiniMax `music-2.6`。
- 文本规划与 prompt refine 使用 Kimi `kimi-k2.5`。
- 验证不是固定打分器，而是 query-conditioned validator framework。
- 系统先用 Kimi 把 query 和 brief 编译成自然语言 checklist。
- 然后把每个 checklist item 路由给对应的 validation skill。
- 每个 validation skill 目录里都带有自己的 `tool.py`，真正执行音频分析。
- 当前 validator skills：
- `semantic_alignment_validator`：CLAP 音频文本语义对齐
- `tempo_checker`：BPM 估计与容差检查
- `rhythm_pattern_checker`：节奏稳定度 / four-on-the-floor 脉冲
- `vocal_presence_checker`：人声存在与性别倾向
- `section_energy_checker`：副歌抬升/后段峰值
- `tone_checker`：warm / dark / bright 等音色倾向
- `mix_health_checker`：时长、响度、静音、削波
- `aesthetic_quality_checker`：Audiobox Aesthetics
- 验证结果除了总分，还会生成 `hard_failures`、`protected_checks`、`next_prompt_guidance`，直接参与下一轮 prompt refine。

## 为什么默认不用 FAD 进主循环

`FAD` 很有价值，但它天然更偏“评估一个生成集合相对于一个参考集合的分布距离”。`fadtk` 的命令也要求 `<baseline> <evaluation-set>` 两个目录输入，所以它适合离线 benchmark，不适合当前这种“单条 query 的逐轮 prompt 优化”主循环。

因此这个 demo 的默认闭环是：

- `checklist compiler` 负责“这次到底要检查什么”，并且由 Kimi 做 checklist decomposition，而不是靠关键词硬匹配
- `validator router` 负责把 checklist item 分配给最合适的 validation skill
- `validation skill/tool.py` 负责真正跑 CLAP、librosa、Audiobox 等工具
- `verifier agent` 负责把 check-level 结果压缩成下一轮可执行的 prompt 改写建议

说明：当前实现默认要求 CLAP 可用。`transformers` 加载 CLAP 权重需要 `torch>=2.6`，所以环境文件已经相应提高了最低版本要求。

如果你后面要做更正式的 benchmark，可以在离线评测阶段额外接入 `FADtk`。

## 参考依据

- MiniMax 官方音乐生成 API：<https://platform.minimaxi.com/docs/api-reference/music-generation>
- Kimi K2.5 官方 API 文档：<https://platform.kimi.ai/docs/guide/kimi-k2-5-quickstart>
- LAION CLAP 官方仓库：<https://github.com/LAION-AI/CLAP>
- Meta Audiobox Aesthetics 官方仓库：<https://github.com/facebookresearch/audiobox-aesthetics>
- Microsoft FADtk 官方仓库：<https://github.com/microsoft/fadtk>

## 目录结构

```text
music-agent-demo/
├── .env.example
├── environment.yml
├── README.md
└── music_agent_demo/
    ├── agent.py
    ├── audio_utils.py
    ├── cli.py
    ├── config.py
    ├── evaluator.py
    ├── llm_client.py
    ├── music_client.py
    ├── schemas.py
    ├── skill_manager.py
    ├── utils.py
    ├── skills/
    └── validation_skills/
        ├── <skill>/SKILL.md
        └── <skill>/tool.py
```

## 环境安装

```bash
cd /home/chang/workspace/26spring/music/music-agent-demo
conda env create -f environment.yml
conda activate music-agent-demo
cp .env.example .env
```

然后把 `.env` 里的 API key 填好。

如果你有 CUDA，可以在该 conda 环境中安装对应 CUDA 版本的 `torch/torchaudio`。

### 可选：安装 Audiobox Aesthetics

```bash
pip install -r requirements-audiobox.txt
```

默认主流程始终包含 CLAP；若要启用 Audiobox，请先安装上述可选依赖并在 `.env` 里设置 `USE_AUDIOBOX=true`。

## 运行

### 1. 只做 skill routing + prompt planning，不调用音乐生成

```bash
python -m music_agent_demo.cli "做一首带城市夜景氛围的 R&B 流行歌，女声，略微忧郁，但副歌要有抬升感" --dry-run
```

### 2. 真实跑 3 轮优化

```bash
python -m music_agent_demo.cli "做一首带城市夜景氛围的 R&B 流行歌，女声，略微忧郁，但副歌要有抬升感" --iterations 3
```

### 3. 强制纯音乐

```bash
python -m music_agent_demo.cli "电影感钢琴与弦乐，孤独、雨夜、缓慢推进" --instrumental --iterations 2
```

## 输出

每次运行会在 `runs/<timestamp-slug>/` 下生成：

- `plan.json`：skill 路由、初始 brief、validation checklist
- `attempt_01/`, `attempt_02/`...：每轮请求、音频、评估结果
- `summary.json`：最佳轮次、分数变化、最终 prompt

## 当前验证链

- `checklist compiler`
- 用 Kimi 对 `query + brief` 做 checklist decomposition，输出自然语言检查句子。
- 例如：`The track should stay around 100 BPM.`、`The chorus should open up clearly from the restrained verse.`

- `validator router`
- 把每个 checklist item 分派给最合适的 validation skill。

- `tool-backed validation skills`
- 每个 validation skill 在自己的 `tool.py` 中执行具体音频分析。
- 例如：`tempo_checker/tool.py` 用 `librosa.beat.beat_track`，`semantic_alignment_validator/tool.py` 用 CLAP，`aesthetic_quality_checker/tool.py` 用 Audiobox。

- `verifier agent`
- 汇总 check-level 结果，输出失败项、已满足项保护、以及下一轮 prompt 改写建议。

这比固定的 `CLAP + heuristics` 打分器更接近 `GEMS` 的思路：不是只问“总分多少”，而是先问“这首歌应该满足哪些自然语言 checklist checks”，再由 skill 内工具去验证。

## 当前简化假设

- 对 vocal music，默认让 MiniMax 用 `lyrics_optimizer=true` 自动补歌词，这样主循环聚焦在 prompt 优化，而不是先把歌词系统做复杂。
- 如果你后面要往 `claude-ai-music-skills` 的方向扩展，可以把歌词 skill、结构 skill、风格 skill 拆得更细。
- 如果你有自己的参考集，可以单独跑 `fadtk` 做离线分布评估，而不是把它放进在线优化 loop。

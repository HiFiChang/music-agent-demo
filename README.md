# music-agent-demo

一个最小可运行的 music generation agent demo。它借鉴 `GEMS` 的核心思路，但把目标从图像改成了音乐：

`user query -> skill routing -> prompt planning -> music generation -> deterministic evaluation -> prompt refinement -> loop`

当前版本追求的是“先跑起来并验证闭环”，而不是完整工程化。为了减少第一次安装失败概率，环境被拆成两层：

- `environment.yml`：核心 agent、Kimi、MiniMax、音频文件处理、heuristics
- `requirements-clap.txt`：推荐安装，负责语义对齐评分
- `requirements-audiobox.txt`：可选安装，负责质量/审美增强评分

## 设计目标

- 和 `GEMS` 一样做循环优化，但只保留单次运行内的历史，不做持久化 memory。
- skill 不是 toolchain，而是 prompt augmentation module。
- 生成端使用 MiniMax `music-2.6`。
- 文本规划与 prompt refine 使用 Kimi `kimi-k2.5`。
- 评估先采用可确定性的自动指标：
- `CLAP`：评估音频和文本意图的语义对齐。
- `Audiobox Aesthetics`：评估内容愉悦度、实用性、制作复杂度、制作质量。
- `Basic heuristics`：时长、静音比例、响度、削波。

## 为什么默认不用 FAD 进主循环

`FAD` 很有价值，但它天然更偏“评估一个生成集合相对于一个参考集合的分布距离”。`fadtk` 的命令也要求 `<baseline> <evaluation-set>` 两个目录输入，所以它适合离线 benchmark，不适合当前这种“单条 query 的逐轮 prompt 优化”主循环。

因此这个 demo 的默认闭环是：

- `CLAP` 负责“像不像我想要的音乐”
- `Audiobox` 负责“音频质量和审美质量怎么样”
- `heuristics` 负责“是不是明显坏样本”

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
    └── skills/
```

## 环境安装

```bash
cd /home/chang/workspace/26spring/music/music-agent-demo
conda env create -f environment.yml
conda activate music-agent-demo
cp .env.example .env
```

然后把 `.env` 里的 API key 填好。

### 推荐：安装 CLAP 评分

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-clap.txt
```

如果你有 CUDA，可以改成 PyTorch 官方对应的 CUDA wheel。

### 可选：安装 Audiobox Aesthetics

```bash
pip install -r requirements-audiobox.txt
```

如果你暂时只想验证闭环和 API 调用，不装这两层也能跑，但评分会退化为 `heuristics-only`。

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

- `plan.json`：skill 路由与初始 brief
- `attempt_01/`, `attempt_02/`...：每轮请求、音频、评估结果
- `summary.json`：最佳轮次、分数变化、最终 prompt

## 当前简化假设

- 对 vocal music，默认让 MiniMax 用 `lyrics_optimizer=true` 自动补歌词，这样主循环聚焦在 prompt 优化，而不是先把歌词系统做复杂。
- 如果你后面要往 `claude-ai-music-skills` 的方向扩展，可以把歌词 skill、结构 skill、风格 skill 拆得更细。
- 如果你有自己的参考集，可以单独跑 `fadtk` 做离线分布评估，而不是把它放进在线优化 loop。

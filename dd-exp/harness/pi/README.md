# Deep-Dream Agent Harness (pi)

把 [pi](https://github.com/earendil-works/pi.git)（MIT）改造成 Deep-Dream 专属
Agent Harness。**不 fork 源码**——以扩展包形式叠加：agent 循环/工具调用/会话
管理由 stock pi 提供（可随上游升级），Deep-Dream 的工作法（工具、skill、
图限定沙箱检索）由本目录定义。MIT 许可保留了未来整体 fork 的自由。

## 组成

| 文件 | 作用 |
|---|---|
| `extensions/deep-dream.ts` | pi 扩展：注册 `dd_scope` / `dd_search` / `dd_ingest` 三个记忆工具（经 `deep-dream --json` CLI，无网络依赖） |
| `models.example.json` | 自定义模型端点模板（OpenAI 兼容，如 kimi-k3） |
| Deep-Dream skill | 复用仓库 `.claude/skills/deep-dream/SKILL.md`（pi 实现 Agent Skills 标准，`--skill` 直载） |

## 安装

```bash
npm i -g @earendil-works/pi-coding-agent

# 扩展：全局或项目级（项目级放 <project>/.pi/extensions/）
mkdir -p ~/.pi/agent/extensions
cp extensions/deep-dream.ts ~/.pi/agent/extensions/

# 模型端点（按需）
cp models.example.json ~/.pi/agent/models.json   # 按模板改

# skill（可选，教 agent 图限定沙箱工作流）
pi --skill /path/to/deep-dream/.claude/skills/deep-dream/SKILL.md
```

## 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `DD_CLI` | CLI 命令（空格分隔，可含参数） | `deep-dream` |
| `DD_CONFIG` | service_config.json 路径 | CLI 默认 |
| `DD_GRAPH` | 目标 graph id | CLI 默认 |
| `DD_TIMEOUT` | 单次 CLI 调用超时（秒） | 300 |

开发态示例：

```bash
DD_CLI=".venv/bin/python -m core.cli" DD_CONFIG=tmp/service_config.smoke.json \
  pi --provider kimi-sz --model kimi-k3
```

## 工作流（图限定沙箱）

论文启示（arxiv 2608.15008）：写深读窄——写入时建图，读取时先窄后深。

1. **`dd_scope(query, materialize=true)`** — 概念图回溯圈出有界文档范围，
   物化成沙箱目录（symlink + manifest，含 episode 偏移）
2. **bash/rg 精读** — 在沙箱内用原生检索精读原文，`manifest.json` 提供每个
   文档的命中概念与偏移锚点
3. （可选）`dd_search` 轻量定位概念；`dd_ingest` 把新文件沉淀进记忆库

## Benchmark 集成

`research/benchmark/pi_track.py` 以 headless 模式（`--mode json`）驱动本
harness 跑 LoCoMo 等数据集，产物与既有 evaluate 轨道同格式。

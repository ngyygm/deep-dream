/**
 * Deep-Dream harness extension for pi.
 *
 * 把 stock pi 变成 Deep-Dream 原生 agent：注册记忆工具（scope 圈范围 /
 * search 概念检索 / ingest 入库），全部经 `deep-dream --json` CLI 实现，
 * 不依赖网络 API。图限定沙箱工作流（graph bounds scope → bash 精读）
 * 见配套 skill（.claude/skills/deep-dream/SKILL.md，pi --skill 直载）。
 *
 * 配置（环境变量）：
 *   DD_CLI     CLI 命令（默认 "deep-dream"；开发态可用
 *              ".venv/bin/python -m core.cli"，空格分隔）
 *   DD_CONFIG  service_config.json 路径（多库/远程端点时设置）
 *   DD_GRAPH   目标 graph id（默认 CLI 自身默认值）
 *   DD_TIMEOUT CLI 超时秒数（默认 300；scope 混合检索含 embedding）
 */
import { execFile } from "node:child_process";
import { env } from "node:process";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

function cliArgs(): string[] {
  const base = (env.DD_CLI || "deep-dream").split(/\s+/).filter(Boolean);
  const args = [...base, "--json"];
  if (env.DD_CONFIG) args.push("--config", env.DD_CONFIG);
  return args;
}

function runCli(sub: string[], timeoutS: number): Promise<string> {
  const parts = cliArgs();
  const cmd = parts[0];
  const args = [...parts.slice(1), ...sub];
  const graph = env.DD_GRAPH;
  if (graph) args.push("--graph", graph);
  return new Promise((resolve) => {
    execFile(
      cmd,
      args,
      { timeout: timeoutS * 1000, maxBuffer: 64 * 1024 * 1024, env },
      (err, stdout, stderr) => {
        if (err) {
          const detail = String(stderr || err.message).slice(0, 2000);
          resolve(`ERROR: ${detail}`);
        } else {
          resolve(String(stdout));
        }
      },
    );
  });
}

function timeout(): number {
  const t = Number(env.DD_TIMEOUT);
  return Number.isFinite(t) && t > 0 ? t : 300;
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "dd_scope",
    label: "Deep-Dream scope",
    description:
      "Deep-Dream 记忆库：为查询圈定有界文档范围（概念图回溯）。返回种子概念、" +
      "命中文档、episode 偏移与片段；materialize=true 时额外把范围物化成沙箱目录" +
      "（symlink + manifest），随后用 bash 在该目录内精读。",
    promptSnippet:
      "dd_scope(query): 在 Deep-Dream 记忆库中圈定与查询相关的文档范围/沙箱",
    promptGuidelines: [
      "回答需要记忆库事实的问题时，先用 dd_scope 圈出文档范围，再用 bash/grep 在范围内精读原文，不要凭空回答。",
      "dd_scope 返回的 episodes 带 start_offset/end_offset，配合沙箱文件可精确定位原文上下文。",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "检索查询（自然语言或关键词）" }),
      mode: Type.Optional(
        Type.String({ description: "bm25 | semantic | hybrid（默认 hybrid）" }),
      ),
      max_docs: Type.Optional(
        Type.Number({ description: "范围文档上限（默认 30）" }),
      ),
      materialize: Type.Optional(
        Type.Boolean({ description: "物化成沙箱目录（默认 false）" }),
      ),
    }),
    async execute(_id, params) {
      const sub = ["scope", params.query];
      if (params.mode) sub.push("--mode", params.mode);
      if (params.max_docs) sub.push("--max-docs", String(params.max_docs));
      if (params.materialize) sub.push("--materialize");
      const out = await runCli(sub, timeout());
      return { content: [{ type: "text", text: out }], details: {} };
    },
  });

  pi.registerTool({
    name: "dd_search",
    label: "Deep-Dream search",
    description:
      "Deep-Dream 记忆库概念检索：按概念/实体名找 family 与相关概念，" +
      "适合先定位再深入（scope 的轻量前置）。",
    promptSnippet: "dd_search(query): Deep-Dream 概念检索",
    promptGuidelines: [
      "只需要定位概念/实体（不需要原文）时用 dd_search，需要原文证据时用 dd_scope。",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "概念/实体查询" }),
      mode: Type.Optional(
        Type.String({ description: "bm25 | semantic | hybrid" }),
      ),
    }),
    async execute(_id, params) {
      const sub = ["find", params.query];
      if (params.mode) sub.push("--mode", params.mode);
      const out = await runCli(sub, timeout());
      return { content: [{ type: "text", text: out }], details: {} };
    },
  });

  pi.registerTool({
    name: "dd_ingest",
    label: "Deep-Dream ingest",
    description:
      "把文件写入 Deep-Dream 记忆库。prose（自然语言，完整 LLM 管线）或 " +
      "log（日志/遥测，零 LLM 快速通道：时间窗切块 + 模式蒸馏）。",
    promptSnippet: "dd_ingest(path, profile): 文件入库（prose/log）",
    promptGuidelines: [
      "用户要求记住某文件或沉淀工作产物时，用 dd_ingest(path) 入库；日志类文本用 profile=log。",
    ],
    parameters: Type.Object({
      path: Type.String({ description: "文件路径（绝对或相对 cwd）" }),
      profile: Type.Optional(
        Type.String({ description: "prose | log（默认 prose）" }),
      ),
      name: Type.Optional(Type.String({ description: "文档标题（默认文件名）" })),
    }),
    async execute(_id, params) {
      const sub = ["ingest", params.path, "--profile", params.profile || "prose"];
      if (params.name) sub.push("--name", params.name);
      const out = await runCli(sub, timeout());
      return { content: [{ type: "text", text: out }], details: {} };
    },
  });
}

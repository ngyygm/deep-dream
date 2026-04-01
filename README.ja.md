<p align="center">
  <img src="https://img.shields.io/github/stars/ngyygm/deep-dream?style=for-the-badge&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/ngyygm/deep-dream?style=for-the-badge&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/license/ngyygm/deep-dream?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Neo4j-5.x-018BFF?style=for-the-badge&logo=neo4j" alt="Neo4j"/>
</p>

<p align="center">
  <img src="docs/images/logo.jpeg" alt="Deep Dream Logo" width="180"/>
</p>

<h1 align="center">🌊 Deep Dream</h1>

<p align="center">
  <em>Agent の全ライフサイクル記憶 — 人間のように記憶し、回溯し、夢を見る</em>
</p>

<p align="center">
  <a href="README.md">🇨🇳 中文</a> · <a href="README.en.md">🇬🇧 English</a> · <a href="README.ja.md">🇯🇵 日本語</a>
</p>

---

<p align="center">
  <img src="docs/images/hero.svg" alt="Deep Dream Hero" width="100%"/>
</p>

> 💤 **人は人生の三分の一を睡眠に費やす。**
>
> これは決して無駄ではない。睡眠中、脳は経験を**再生**し、断片を**再構成**し、覚醒時には気づかなかった隠れたつながりを**発見**している。
> レム睡眠のたびに、散らばった断片がネットワークに編まれ、曖昧な直感が洞察として結晶化する。
>
> **Deep Dream は AI Agent に同じ能力を与える。**

---

## ✨ 三つのコア機能

<table>
<tr>
<td width="33%" align="center"><b>🧠 Remember</b><br/>覚醒時の書き込み</td>
<td width="33%" align="center"><b>🔍 Find</b><br/>必要時の検索</td>
<td width="33%" align="center"><b>💭 Dream</b><br/>睡眠時の統合</td>
</tr>
<tr>
<td>

テキスト→実体<br/>
文書→関係<br/>
バージョン管理

</td>
<td>

意味検索<br/>
グラフ拡張<br/>
時間遡行

</td>
<td>

自律戦略選択<br/>
ツール呼び出し<br/>
関係発見

</td>
</tr>
</table>

<p align="center">
  <img src="docs/images/architecture.jpeg" alt="Deep Dream Architecture" width="650"/>
</p>

---

## 🤔 なぜ Agent は夢を見る必要があるのか？

| 🧑 人間の記憶 | 🤖 Deep Dream |
|:---:|:---:|
| 日中の経験 → 記憶の書き込み | テキスト/文書 → **Remember** で知識グラフに |
| 過去を思い出す → 記憶の検索 | 自然言語質問 → **Find** で意味検索 |
| 夜の睡眠 → 記憶の統合 | Dream Agent → **DeepDream** で新関係を自律発見 |

従来の知識グラフは**静的**——書いたものがそのまま。DeepDream は Agent に同じ能力を与える：

- 🌉 **意味的距離を超える** — 類似実体だけでなく、遠い意味的距離の接続も発見
- 🦘 **跳躍的思考** — 夢の中の自由連想のように、無関係に見える概念間を跳躍
- 🔄 **マルチ戦略** — 8種の戦略を循環し、連想・対比・時間・クロスドメインをカバー
- ♾️ **永遠に続く** — Agent が「睡眠」中、夢は無限の反復で持続

> ⚠️ **重要な制約:** Dream Agent は**既存の実体間の新関係のみ発見**し、存在しない実体を捏造しない。人間が夢の中で既存の記憶を再編成するのと同じ。すべての夢の発見には `source: dream` マークが付与される。

---

## 🏗️ コアアーキテクチャ

```
Remember（覚醒時）         Find（必要時）          Dream（睡眠時）
┌──────────────┐     ┌──────────────┐     ┌────────────────────┐
│ 📝 テキスト→実体│     │ 🔍 意味検索   │     │ 💭 Dream Agent     │
│ 📄 文書→関係  │     │ 🕸️ グラフ拡張 │     │   ├─ 戦略選択       │
│ 📦 バージョン │     │ ⏳ 時間遡行   │     │   ├─ LLM 計画       │
│   書き込み    │     │              │     │   ├─ ツール実行     │
└──────┬───────┘     └──────┬───────┘     │   ├─ 観察・反思     │
       │                    │              │   └─ 関係保存       │
       ▼                    ▼              └────────┬───────────┘
   ┌───────────────────────────────────────────────────▼─────────┐
   │                 🧬 統一記憶知識グラフ                         │
   │    Entity バージョン鎖 · Relation バージョン鎖 · Episode      │
   └──────────────────────────────────────────────────────────────┘
```

Dream Agent はハードコードされたループではなく、**自律的エージェント** — ツール呼び出しループで自律的に判断する：
1. 📋 どの戦略でシード実体を取得するか
2. 🔭 どの実体と関係を巡回・観察するか
3. 💡 いつ新しい関係仮説を提案するか
4. 📝 いつ夢の発見を記録するか

---

## 🚀 クイックスタート

### インストール

```bash
git clone https://github.com/ngyygm/deep-dream.git
cd deep-dream
pip install -r requirements.txt
cp service_config.example.json service_config.json
# service_config.json を編集: LLM と Embedding を設定
python -m server.api --config service_config.json
```

ブラウザで **http://127.0.0.1:16200/** を開く 🎉

### 📝 記憶の書き込み

```bash
curl -s -X POST http://localhost:16200/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"text":"林嘿嘿は考古学博士で、洞窟で話す白狐に出会った。白狐は洞窟を300年守っていると言った。","event_time":"2026-03-09T14:00:00"}'
```

### 🔍 記憶の検索

```bash
curl -s -X POST http://localhost:16200/api/v1/find \
  -H "Content-Type: application/json" \
  -d '{"query": "林嘿嘿と白狐のあいだに何があったか"}'
```

### 💭 夢境統合の開始

```bash
curl -s -X POST http://localhost:16200/api/v1/find/dream/agent/start \
  -H "Content-Type: application/json" \
  -d '{
    "max_cycles": 10,
    "strategies": ["free_association", "cross_domain", "leap"],
    "strategy_mode": "round_robin",
    "confidence_threshold": 0.6
  }'
```

---

## 🌈 8つの夢戦略

| 戦略 | 🎭 アナロジー | 🎯 目標 |
|------|-------------|---------|
| `free_association` | 🔗 自由連想 | ランダム実体間の隠れた接続 |
| `contrastive` | ⚖️ 対比分析 | 類似実体間の差異と対比 |
| `temporal_bridge` | ⏳ タイムトラベル | 時間を超えた進化パターン |
| `cross_domain` | 🌉 異分野インスピレーション | 異分野の意外な架け橋 |
| `orphan_adoption` | 🏠 孤立救済 | 孤立実体のつながり発見 |
| `hub_remix` | 🔀 ハブ再結合 | 中核ノード間の新パス |
| `leap` | 🦘 創造的跳躍 | 遠距離の連想ジャンプ |
| `narrative` | 📖 物語紡ぎ | 断片を物語に紡ぐ |

---

## 🛠️ Dream Agent ツールボックス

Dream Agent は8つのツールで知識グラフと対話。LLM が自律的にツールを選択：

| ツール | 📌 用途 |
|--------|---------|
| `get_seeds` | 戦略に基づきシード実体を取得（出発点） |
| `get_entity` | 実体の詳細と直接関係を表示 |
| `traverse` | BFS 多数ホップ拡張で隣接を発見 |
| `search_similar` | 意味類似度検索 |
| `search_bm25` | BM25 キーワード検索 |
| `get_community` | コミュニティとメンバーを取得 |
| `create_relation` | 夢で発見した関係を保存 |
| `create_episode` | 夢周期の発見を記録 |

---

## 📋 API リファレンス

### Dream Agent

```
POST /api/v1/find/dream/agent/start
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `max_cycles` | int | 10 | 夢周期数 (1-50) |
| `strategies` | string[] | `["free_association","cross_domain","leap"]` | 使用する戦略 |
| `strategy_mode` | string | `"round_robin"` | モード: `round_robin` / `random` / `adaptive` |
| `confidence_threshold` | float | 0.6 | 関係保存の最低信頼度 |
| `max_tool_calls_per_cycle` | int | 15 | 周期あたりの最大ツール呼び出し |

### 記憶操作

| エンドポイント | 説明 |
|--------------|------|
| `POST /api/v1/remember` | 記憶の書き込み（非同期） |
| `POST /api/v1/find` | 統合意味検索 |
| `POST /api/v1/find/traverse` | BFS グラフ巡回 |
| `GET /api/v1/find/entities` | 実体一覧/検索 |
| `GET /api/v1/find/relations` | 関係一覧/検索 |
| `GET /api/v1/find/snapshot` | タイムトラベルスナップショット |
| `POST /api/v1/find/ask` | Agent メタクエリ（自然言語） |

---

## ⚙️ 設定

`service_config.example.json` を参照：

| 設定 | 説明 |
|------|------|
| `host` / `port` | サービスアドレス、デフォルト `0.0.0.0:16200` |
| `storage.backend` | バックエンド: `"sqlite"` / `"neo4j"` |
| `llm` | LLM 設定 (Ollama / OpenAI 互換 / GLM など) |
| `embedding` | Embedding モデル（ローカルパスまたは HuggingFace 名） |
| `dream_llm` | 夢専用 LLM（軽量モデルを別途設定可能） |
| `chunking` | スライディングウィンドウサイズとオーバーラップ |
| `runtime.concurrency.*` | 3層並行制御 |

---

## 🔌 Agent 統合

Deep Dream はスキルを提供し、スキル呼び出しをサポートする任意の Agent（Cursor、Claude Codeなど）が記憶と夢機能を直接利用可能：

- **スキル名**: `deep-dream`
- **パス**: `.claude/skills/deep-dream/`
- **トリガー**: `"开始做梦"` / `"dream"` / `"深度复习"`
- **統合方法**: スキルを Agent のスキルディレクトリに追加

---

## 🧪 技術スタック

| レイヤー | 技術 |
|----------|------|
| グラフDB | Neo4j 5.x Community |
| ベクトル検索 | sqlite-vec (ANN KNN) |
| LLM | OpenAI 互換プロトコル (GLM / Ollama / LM Studio) |
| Embedding | ローカルモデル / HuggingFace |
| Web | Flask + ネイティブ SPA ダッシュボード |
| Agent パターン | Tool-based Agent Loop (claude-code-rev にインスパイア) |

---

## 📄 License

[LICENSE](LICENSE) を参照。

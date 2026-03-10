<p align="center">
  <img src="https://img.shields.io/github/stars/ngyygm/Temporal_Memory_Graph?style=for-the-badge&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/ngyygm/Temporal_Memory_Graph?style=for-the-badge&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/license/ngyygm/Temporal_Memory_Graph?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python" alt="Python"/>
</p>

<p align="center">
  <strong>Temporal Memory Graph (TMG)</strong>
</p>
<p align="center">
  <b>Agent 向け長期記憶</b> —— 人間のように記憶し、想起し、時間を遡る。
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a>
</p>

---

## 概要

TMG は AI Agent に**時間付きの自然言語記憶**を提供します。Agent 向けの**長期記憶の格納・検索**、**人間と同様の**自然言語による記憶・想起、そして**時間を第一級**として扱い、各記憶は追跡可能、エンティティ・関係はバージョン鎖を持ちます。経験は単一の統一グラフに書き込まれ、自然言語の質問で関連領域を喚起し、「あのとき何があったか」のような時間遡行をサポートします。

| 方針 | 説明 |
|------|------|
| **Agent 向け** | エージェント用の長期記憶の読み書きであり、人間向けメモやナレッジベースではない。 |
| **人間のように** | 自然言語で書き、自然言語で検索。事前定義タグに依存せず、システムが概念抽出・関係構築を行う。 |
| **時間は第一級** | 記憶にタイムスタンプ、エンティティ・関係にバージョン鎖。時間範囲・時点での遡行が可能。 |
| **統一グラフ** | 全記憶を一つのグラフに格納。意味検索＋グラフ拡張で「関連記憶の領域」を返す。 |

システムの責務：**Remember**（書き込み）と **Find**（検索）のみ。**Select**（何をどう使うか）は呼び出し側が担当。

### 従来の知識グラフとの比較

| 観点 | 従来の KG | TMG |
|------|-----------|-----|
| 関係 | 固定タイプ（is_a, located_in 等） | 自然言語記述（概念辺） |
| 書き込み | 構造化入力とスキーマが必要 | テキスト/文書をそのまま投入、システムが抽出・整合 |
| 時間 | 静的または単純なタイムスタンプ | バージョン鎖＋タイムスタンプ、時間遡行クエリ対応 |
| 更新 | 上書きが多い | 追記型、履歴を保持 |
| 検索 | 構造化クエリ・タグフィルタ | 意味検索＋グラフ近傍拡張 |

---

## アーキテクチャ

```mermaid
flowchart TB
    subgraph Input["入力"]
        T[テキスト / 文書]
        F[ファイルアップロード]
    end

    subgraph Pipeline["記憶パイプライン"]
        W[スライディングウィンドウ]
        M[Memory Agent]
        M --> M1[記憶キャッシュ更新]
        M --> M2[概念エンティティ抽出]
        M --> M3[概念関係抽出]
        M --> M4[グラフ意味整合]
        M --> M5[バージョン書き込み]
    end

    subgraph Storage["統一記憶グラフ"]
        E[(Entity バージョン鎖)]
        R[(Relation バージョン鎖)]
        C[(MemoryCache)]
    end

    subgraph Find["検索"]
        Q[自然言語クエリ]
        S[意味召回]
        G[グラフ拡張]
        Tf[時間フィルタ]
        Out[局所記憶領域]
    end

    T --> W
    F --> W
    W --> M
    M --> E
    M --> R
    M --> C
    Q --> S
    S --> G
    G --> Tf
    Tf --> Out
    E -.-> S
    R -.-> S
```

---

## クイックスタート

```bash
cp service_config.example.json service_config.json
# service_config.json を編集：LLM と embedding
python service_api.py --config service_config.json
```

**記憶の書き込み：**

```bash
curl -s -X POST http://localhost:16200/api/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "林嘿嘿は考古学博士で、洞窟で話す白狐に出会った。白狐は三百年この洞窟を守ってきたと言った。"}' | jq
```

**記憶の検索：**

```bash
curl -s -X POST http://localhost:16200/api/find \
  -H "Content-Type: application/json" \
  -d '{"query": "林嘿嘿と白狐のあいだに何があったか"}' | jq
```

---

## Skill の利用（Agent 連携）

TMG は **Skill** を同梱しており、Cursor や Claude などの Agent がドキュメントに従ってデプロイ・設定・起動・API 呼び出しを行えます（HTTP クライアントの手書き不要）。

### Skill の場所と内容

- **パス:** `Temporal_Memory_Graph/skills/tmg-memory-graph/`
- **ファイル:** `SKILL.md`（Agent 向け手順）、`reference.md`（API クイックリファレンス）
- **役割:** 「ドキュメントを読んで実行」できる Agent が SKILL を読むことで、TMG をいつ使うか・どうデプロイするか・どう API を呼ぶかを実行可能にする。

### Agent に TMG を使わせる 3 ステップ

1. **Skill を Agent に渡す**  
   - **Cursor:** ルールに「TMG 記憶を使うときは `Temporal_Memory_Graph/skills/tmg-memory-graph/SKILL.md` を読み従う」と追記するか、要点を `.cursor/rules` に書く。  
   - **Claude / その他:** `skills/tmg-memory-graph/` をその Agent のスキルディレクトリやナレッジベースに追加する。

2. **自然言語でトリガー**  
   「これを覚えておいて」「〇〇について以前覚えたことを検索して」「TMG 記憶サービスに接続して」などとユーザーが言ったときに、Agent が SKILL を読み、フロー（サービス確認 → remember/find）を実行する。

3. **Agent が行うこと**  
   - サービスが未起動なら: リポジトリ clone → `service_config.json` 設定 → `python service_api.py` 起動 → `GET /health` で確認。  
   - 記憶: `POST /api/remember`（本文 `text` または `file_path`、あるいは multipart アップロード）。  
   - 検索: `POST /api/find` に自然言語の `query`。必要に応じてエンティティ/関係/バージョン/サブグラフの原子 API を使用。

---

## API 概要

### Remember — 書き込み

| 方式 | 説明 |
|------|------|
| テキスト | JSON body: `{"text": "..."}` |
| ローカルファイル | JSON body: `{"file_path": "/path/to/file"}`（サーバ側パス） |
| アップロード | multipart: `file=@/path/to/file`（txt / md / pdf / docx） |

省略可能: `source_name`, `load_cache_memory`。内部で切片化・記憶キャッシュ更新・エンティティ/関係抽出・グラフ整合・バージョン書き込みを実行。

### Find — 検索

- **推奨:** `POST /api/find` — 意味召回・グラフ拡張・時間フィルタを 1 リクエストで実行。必須は `query`、他は任意。  
- **原子 API:** エンティティ検索（`/api/find/entities/search` 等）、関係、記憶キャッシュ、サブグラフ作成/拡張/フィルタ、統計（`/api/find/stats`）。  

完全なパスとパラメータは `skills/tmg-memory-graph/reference.md` および `service_api.py` を参照。

### レスポンス形式

- 成功: `{"success": true, "data": ..., "elapsed_ms": 123.45}`
- 失敗: `{"success": false, "error": "メッセージ", "elapsed_ms": 12.34}`

---

## データモデル（概要）

- **Entity:** 概念エンティティ。`entity_id`（論理 ID）、`id`（バージョン絶対 ID）、`name`、`content`（自然言語）、`physical_time`。複数バージョンで鎖を形成。  
- **Relation:** 概念関係。自然言語記述（固定関係タイプではない）。`entity1/2_absolute_id`、`physical_time`、バージョン鎖。  
- **MemoryCache:** システム内部のコンテキスト要約鎖。整合・推論に使用。  

内容はすべて自然言語＋時間。事前定義タグ体系はなし。

---

## 設定

`service_config.example.json` を参照し、`service_config.json` で以下を設定：

- **サービス:** `host`, `port`, `storage_path`  
- **LLM:** `api_key`, `model`, `base_url`, `think`  
- **Embedding:** `embedding.model`（ローカルパスまたは HuggingFace 名）、`embedding.device`  
- **チャンク:** `chunking.window_size`, `chunking.overlap`  
- **サブグラフ:** `subgraph_max_count`, `subgraph_ttl_seconds`  

---

## License

リポジトリルートの [LICENSE](LICENSE) を参照（存在する場合）。

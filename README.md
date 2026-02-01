# Brontes: Requirements-Driven AI Agent Verification Project

**「実現したいこと」を与えると、それを実現する Python 関数・仕様・判断の背景が一体で得られる**

---

## 🎯 このプロジェクトは何をするのか

Brontes は、以下の流れで成果物を生成します：

1. **要件を与える**：「実現したいこと」を自然言語で記述
2. **エージェントが探索する**：Python REPLで試行錯誤・実装を模索
3. **成果物が残る**：実装（Python関数）・仕様・判断の背景が一体で抽出される

---

## 📝 要件とは何か

**要件 = 実現したいこと**

厳密な仕様書ではなく、探索を開始するための入力です。

例：
```
2025年下半期のAWSニュースから、生成AI関連かつ
東京リージョン対応の記事を自動抽出する関数を作りたい
```

エージェントはこの要件を受けて、「どう実装すべきか」を探索します。

---

## 📦 生成される成果物

要件を与えると、以下が一体で得られます：

### 1. Python 関数（How：実装）

```python
def filter_aws_news(xml_path, start_date, end_date, keywords, region):
    """
    AWS What's New から記事をフィルタリング
    
    Args:
        xml_path: RSS XMLファイルのパス
        start_date: 開始日（ISO形式）
        end_date: 終了日（ISO形式）
        keywords: 検索キーワードのリスト
        region: 対象リージョン (例: "ap-northeast-1")
    
    Returns:
        フィルタ後の記事リスト
        
    実装根拠：
    - XML解析にはxml.etree.ElementTreeを使用（標準ライブラリで十分）
    - 日付判定はdatetimeで比較（タイムゾーンを考慮）
    - キーワードは大文字小文字を区別しない部分一致
    """
    # 実装コード...
```

### 2. 仕様（Spec）

docstring・引数・戻り値から読み取れる関数の仕様

### 3. 判断の背景（Why）

- なぜその実装を選んだか
- どんな試行錯誤を経たか
- 失敗した探索も含めて記録

**重要な点**：How / Spec / Why は別々に管理されるのではなく、**同一の実行トレースから一体で観測・抽出される**

---

## 💡 なぜ Python 関数として残すのか

Python関数という形で残すことで：

- **再実行可能**：2回目以降はLLM不要で高速実行
- **検証可能**：単体で動作確認・テスト可能
- **修正可能**：探索結果をベースに人間が改良できる
- **実行単位として残る**：思考・探索の結果が、そのまま実行可能な形で残る

---

## 🧪 実証実験：AWS 記事フィルタリング

### 与えた要件

```
2025年下半期のAWSニュース（XML形式）から、
生成AI関連かつ東京リージョン対応の記事を
自動抽出する関数を開発したい
```

### 生成された成果物

**1. トレースファイル** ([outputs/trace.jsonl](outputs/trace.jsonl))  
エージェントの全思考・実行・失敗を含む完全な記録

**2. ノートブック** ([outputs/agent_replay.ipynb](outputs/agent_replay.ipynb))  
探索プロセスを対話的に追跡・検証できる形式

**3. Python関数**  
実装・仕様・設計根拠が一体となった関数（ノートブックから抽出可能）

---

## 🚀 実行方法

```bash
# エージェント実行
uv run python src/main.py
```

実行により：
1. エージェントが Python REPL で問題解決を探索
2. 思考・実行・結果が `outputs/trace.jsonl` に記録
3. ノートブック `outputs/agent_replay.ipynb` が自動生成
4. Python関数が抽出可能な状態になる

---

## 📋 プロジェクト構成

```
brontes/
├── src/
│   ├── main.py                  # エージェント実行
│   └── generate_notebook.py     # トレース→ノートブック変換
├── outputs/
│   ├── trace.jsonl              # 実行トレース
│   └── agent_replay.ipynb       # 探索プロセス
└── data/                        # 入力データ
```

---

## 🔧 技術構成

- **Strands Agents**: エージェントフレームワーク
- **AWS Bedrock** (Claude Haiku 4.5): LLM
- **Python REPL**: コード実行環境
- **OpenTelemetry**: トレース記録

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

**Brontes**: AIエージェントの探索と人間の問題解決を結ぶ試み

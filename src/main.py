"""Brontes: AWS ニュース記事フィルタリングエージェント.

このモジュールは、Strands Agentsを使用してAWSニュースをフィルタリングするエージェント
アプリケーションの主要な実行エントリーポイントです。

エージェントの処理フロー：
1. ユーザーのリクエストを解析
2. python_replツールを使用してPythonコードを実行
3. XMLファイルからAWSニュース記事をフィルタリング
4. 実行トレースをJSONL形式で記録
5. トレースからJupyterノートブックを生成
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
from strands_tools import python_repl

from generate_notebook import (
    create_notebook,
    extract_agent_timeline,
    extract_python_repl_code,
    extract_trace_metadata,
)

SYSTEM_PROMPT = """## タスク
指定された条件に基づいて、XMLファイルからAWSニュース記事を抽出する。

## データソース
- **パス**: `data/01_raw/*.xml`
- **形式**: RSS 2.0 (各 `<item>` が1記事)
- **フィールド**:
  - `<guid>`: 記事ID
  - `<title>`: タイトル
  - `<pubDate>`: 公開日時（RFC 2822形式）
  - `<description>`: 記事本文
  - `<link>`: URL

## 入力条件
- **期間**: 開始日時 ≤ pubDate ≤ 終了日時
- **トピックキーワード**: リスト形式で提供
- **リージョンキーワード**: リスト形式で提供

## 出力形式
```json
{
  "articles": [
    {
      "guid": "記事ID",
      "title": "タイトル",
      "pubDate": "ISO 8601形式の日時",
      "description": "本文",
      "link": "URL"
    }
  ],
  "summary": {
    "total_count": 総記事数,
    "filtered_count": フィルタ後記事数,
    "from_date": "ISO 8601形式",
    "to_date": "ISO 8601形式"
  }
}
```

## 抽出条件
1. **期間**: 指定された期間内に公開された記事
2. **キーワード**: title または description に以下を含む
   - トピックキーワードから1つ以上 **かつ**
   - リージョンキーワードから1つ以上
   - 大文字小文字を区別しない
3. **重複排除**: guid でユニーク化
4. **ソート**: pubDate 降順（新しい順）

## 制約
- 利用可能なツール：python_repl のみ
- 最終結果は再利用可能な単一の関数で取得すること
- 関数内で値をハードコードしない（全てパラメータで受け取る）
- グローバル変数に依存しない
- ファイル出力しない

## コメント要件
- 探索に使用するPythonコードには、処理の意図や確認内容を説明するコメントを含めること
- 最終関数には、引数・戻り値・処理フローを説明する適切なdocstringとコメントを含めること
- コメントは必要十分な情報を簡潔に記述すること

## 実装フロー
- Pythonコードは作成して実行するのみ
- 追加の説明は不要（コード内のコメント・docstringで十分）
- **重要**: 実装仕様の確認や説明用のPythonコード（print文で仕様表示など）は作成しない
- 探索と最終関数取得のみに集中すること

## 技術的要件
最終関数の直前に実装の設計判断とその根拠を説明するコメントブロックを出力し、その後に最終関数を出力すること。さらに、最終関数を呼び出して結果を取得するコードも出力すること。以下のマーカー形式を使用（トレースからの自動抽出のため）：

```python
# === IMPLEMENTATION_RATIONALE_START ===
# [この実装に至った設計判断とその根拠を簡潔に説明]
# 例：
# - ライブラリ選択: ○○を使用（理由：性能/標準ライブラリ/可読性など）
# - データ構造: ○○を選択（理由：検索効率/メモリ効率など）
# - アルゴリズム: ○○の順序で処理（理由：効率性/正確性など）
# - エラー処理: ○○の方針（理由：堅牢性/デバッグ容易性など）
# === IMPLEMENTATION_RATIONALE_END ===

# === FINAL_FUNCTION_START ===
# 必要な全てのimport文を含める
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

def 関数名(...):
    \"\"\"
    関数の説明
    
    Args:
        引数の説明
    
    Returns:
        戻り値の説明
    \"\"\"
    # 処理の説明コメント
    ...
# === FINAL_FUNCTION_END ===

# === FUNCTION_CALL_START ===
# パラメータの設定（必要に応じて外部変数として定義）
data_dir_path = Path('data/01_raw')
from_date = datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
to_date = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
topic_keywords = ['Bedrock', 'SageMaker', ...]
region_keywords = ['Tokyo', 'ap-northeast-1', ...]

# 関数の実行
result = 関数名(
    data_dir=data_dir_path,
    from_date=from_date,
    to_date=to_date,
    topic_keywords=topic_keywords,
    region_keywords=region_keywords,
)
print(result)
# === FUNCTION_CALL_END ===
```
"""  # noqa: E501

USER_PROMPT = """以下の条件でAWSニュース記事をフィルタリングしてください：
**期間**: 2025年下半期（7月1日 00:00:00 ～ 12月31日 23:59:59）
**トピック**: 生成AI関連のアナウンスと更新で東京リージョンで利用可能なもの

**キーワード**:
- 生成AI関連: Bedrock, SageMaker, Claude, LLM, foundation model, generative AI, embedding, fine-tuning, inference, RAG, prompt, model training, neural network, transformer, deep learning, machine learning
- 東京リージョン関連: Tokyo, ap-northeast-1, Asia Pacific (Tokyo), available in Tokyo, Tokyo region, all regions, all aws regions,
"""  # noqa: E501


def main() -> None:
    """エージェントアプリケーションの実行エントリーポイント.
    
    このメイン関数は以下の処理を実行します：
    1. 出力ディレクトリの作成と管理
    2. OpenTelemetryトレーシングの設定（trace.jsonl に記録）
    3. Bedrockモデルの初期化
    4. エージェントの生成とプロンプト実行
    5. 実行トレースからJupyterノートブックの自動生成
    
    出力ファイル：
    - outputs/trace.jsonl: 完全な実行トレース（OpenTelemetry形式）
    - outputs/agent_replay.ipynb: 再現可能なノートブック（思考プロセス+最終関数付き）
    """
    # 出力ディレクトリの準備
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    notebook_path = output_dir / "agent_replay.ipynb"

    # OpenTelemetryトレーシングの設定
    # エージェントの全ての操作（プロンプト、ツール実行、レスポンス）をJSONL形式で記録
    strands_telemetry = StrandsTelemetry()
    with trace_path.open("w", encoding="utf-8") as trace_file:
        strands_telemetry.setup_console_exporter(
            out=trace_file,
            formatter=lambda span: span.to_json() + "\n",
        )

        # LLMモデルの初期化（Bedrock Claude Haiku）
        bedrock_model = BedrockModel(
            # model_id="jp.amazon.nova-2-lite-v1:0",
            model_id="jp.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        # エージェントの初期化とタスク実行
        # system_prompt: エージェントの役割と制約条件
        # tools: 利用可能なツール（python_repl のみ）
        # user_prompt: ユーザーの具体的なリクエスト
        agent = Agent(
            model=bedrock_model,
            tools=[python_repl],
            system_prompt=SYSTEM_PROMPT,
        )

        # エージェントによるタスク実行
        # エージェントはpython_replツールを使用して、
        # XMLファイルからAWSニュースをフィルタリングする処理を実装
        result = agent(USER_PROMPT)
        print(result)

        # 結果の構造化出力（JSON形式）を表示
        import json
        import re

        result_text = str(result)
        # レスポンスからJSON抽出
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            try:
                output_json = json.loads(json_match.group())
                print("\n=== 構造化出力 ===")
                print(json.dumps(output_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                pass

    # トレースからJupyterノートブックを生成
    # トレースには、エージェントの思考プロセスと実行したコードが記録されている
    metadata = extract_trace_metadata(trace_path)
    codes = extract_python_repl_code(trace_path)
    timeline = extract_agent_timeline(trace_path)
    create_notebook(codes, notebook_path, metadata, timeline)

    print(f"\n✅ Notebook created: {notebook_path}")
    print(f"   Extracted {len(codes)} code block(s) from trace")
    print(f"   Timeline events: {len(timeline)}")


if __name__ == "__main__":
    main()

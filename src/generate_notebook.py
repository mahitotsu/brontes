"""Generate a Jupyter Notebook from Strands Agent execution trace.

このモジュールは、Strands AgentのOpenTelemetryトレース（trace.jsonl）から
以下の情報を抽出して、実行可能なJupyterノートブックを生成します：

1. **エージェントのメタデータ**: ユーザープロンプト、システムプロンプト、モデル情報
2. **実行タイムライン**: エージェントの思考プロセスとコード実行の時系列記録
3. **抽出コード**: python_replツールで実行されたPythonコード
4. **再利用可能関数**: トレースから自動抽出された、再利用可能な関数

生成されるノートブックは、エージェントの推論過程を完全に追跡可能にしながら、
最終的に得られた成果物（関数）を簡単に再実行できるようになっています。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Optional

import nbformat as nbf


def extract_trace_metadata(trace_path: Path) -> dict:
    """トレースファイルからメタデータを抽出.
    
    OpenTelemetryトレース（JSONL形式）から以下を抽出します：
    - user_prompt: ユーザーが指定したタスク説明
    - system_prompt: エージェントの役割と制約条件
    - model_id: 使用されたLLMモデルのID
    
    Args:
        trace_path: trace.jsonlファイルのパス
    
    Returns:
        辞書形式のメタデータ：
        {
            "user_prompt": str or None,
            "system_prompt": str or None,
            "model_id": str or None
        }
    """
    metadata = {
        "user_prompt": None,
        "system_prompt": None,
        "model_id": None,
    }

    # JSONL形式のトレースを行単位で読み込み
    # （複数のJSON オブジェクトが改行で区切られている）
    text = trace_path.read_text(encoding="utf-8")
    json_objects = []
    current_obj = ""
    brace_count = 0

    # ブレース（{}）のカウントを使用して個別のJSONオブジェクトを抽出
    for char in text:
        current_obj += char
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and current_obj.strip():
                json_objects.append(current_obj.strip())
                current_obj = ""

    # 各JSONオブジェクトをパースしてメタデータを抽出
    for json_str in json_objects:
        try:
            span = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        # Spanの属性からシステムプロンプトとモデル情報を抽出
        attrs = span.get("attributes", {})
        if not metadata["system_prompt"] and "system_prompt" in attrs:
            metadata["system_prompt"] = attrs["system_prompt"]
        if not metadata["model_id"] and "gen_ai.request.model" in attrs:
            metadata["model_id"] = attrs["gen_ai.request.model"]

        # イベント内からユーザープロンプトと最終レスポンスを抽出
        events = span.get("events", [])
        for event in events:
            event_name = event.get("name", "")
            event_attrs = event.get("attributes", {})

            # ユーザープロンプトを抽出（初出のみ）
            if event_name == "gen_ai.user.message" and not metadata["user_prompt"]:
                content_str = event_attrs.get("content", "")
                try:
                    content_items = json.loads(content_str)
                    for item in content_items:
                        if "text" in item:
                            metadata["user_prompt"] = item["text"]
                            break
                except:  # noqa: E722
                    pass

    return metadata


def _extract_import_lines(source: str) -> list[str]:
    """ソースコードからimport文を抽出.
    
    ASTを使用してPythonソースコードを解析し、
    すべてのimport文を元の順序で抽出します。
    
    Args:
        source: Pythonソースコード文字列
    
    Returns:
        import文のリスト（重複排除済み）
    """
    imports: list[str] = []
    seen: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            line = ast.get_source_segment(source, node)
            if line and line not in seen:
                imports.append(line)
                seen.add(line)

    return imports


def _extract_marked_block(
    codes: list[str],
    start_marker: str,
    end_marker: str,
) -> Optional[tuple[str, int]]:
    """コードブロック内の明示的マーカー間からコードを抽出.
    
    システムプロンプトで指定された形式のマーカー
    （例: # === IMPLEMENTATION_RATIONALE_START ===）で囲まれた
    コードを抽出します。
    
    Args:
        codes: コードブロックのリスト
        start_marker: 開始マーカー文字列
        end_marker: 終了マーカー文字列
    
    Returns:
        (抽出されたコード, ブロックのインデックス) のタプル、
        または見つからない場合はNone
    """
    for idx in range(len(codes) - 1, -1, -1):
        code = codes[idx]
        if start_marker in code and end_marker in code:
            start_idx = code.find(start_marker) + len(start_marker)
            end_idx = code.find(end_marker, start_idx)
            if end_idx > start_idx:
                extracted = code[start_idx:end_idx].strip()
                if extracted:
                    return extracted, idx
    return None


def _extract_implementation_rationale(codes: list[str]) -> Optional[str]:
    """マーク済みブロックから実装の根拠説明を抽出.
    
    システムプロンプトのマーカー形式
    （# === IMPLEMENTATION_RATIONALE_START/END ===）で囲まれた
    コメントブロックを抽出し、コメント記号を除去して
    テキストとして返します。
    
    Args:
        codes: コードブロックのリスト
    
    Returns:
        整形済みの根拠説明テキスト、または見つからない場合はNone
    """
    rationale_marker_start = "# === IMPLEMENTATION_RATIONALE_START ==="
    rationale_marker_end = "# === IMPLEMENTATION_RATIONALE_END ==="

    for code in codes:
        if rationale_marker_start in code and rationale_marker_end in code:
            start_idx = code.find(rationale_marker_start) + len(rationale_marker_start)
            end_idx = code.find(rationale_marker_end, start_idx)
            if end_idx > start_idx:
                # マーカー間のテキストを抽出
                rationale_block = code[start_idx:end_idx].strip()
                # 各行のコメント記号（#）を除去
                lines = rationale_block.split("\n")
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("# "):
                        cleaned_lines.append(line[2:])
                    elif line.startswith("#"):
                        cleaned_lines.append(line[1:].lstrip())
                    elif line:
                        cleaned_lines.append(line)

                rationale_text = "\n".join(cleaned_lines).strip()
                if rationale_text:
                    return rationale_text

    return None


def build_reusable_function(codes: list[str]) -> Optional[tuple[str, dict]]:
    """トレースから再利用可能な関数を構築.
    
    複数のコードブロックから、FINAL_FUNCTION_START/ENDマーカーで
    囲まれた関数定義を抽出し、必要なimport文を追加します。
    
    処理フロー：
    1. 全コードブロックのimport文を収集
    2. FINAL_FUNCTION_START/ENDマーカー間のコード抽出
    3. 不足しているimport文を追加
    
    Args:
        codes: コードブロックのリスト
    
    Returns:
        (関数コード, メタデータ) のタプル、または抽出失敗時はNone
        メタデータ: {
            "extraction_mode": 抽出方法,
            "source_blocks": コードブロック数,
            "source_block_index": 抽出元ブロックのインデックス,
            "imports": import文数
        }
    """
    combined_source = "\n\n".join(codes)
    all_imports = _extract_import_lines(combined_source)

    marked_block = _extract_marked_block(
        codes,
        "# === FINAL_FUNCTION_START ===",
        "# === FINAL_FUNCTION_END ===",
    )
    if not marked_block:
        return None

    function_code, block_index = marked_block
    existing_imports = _extract_import_lines(function_code)

    # 全コードブロックから不足しているimport文を追加
    missing_imports = [imp for imp in all_imports if imp not in existing_imports]
    if missing_imports:
        function_code = "\n".join(missing_imports) + "\n\n" + function_code

    summary = {
        "extraction_mode": "marker",
        "source_blocks": len(codes),
        "source_block_index": block_index + 1,
        "imports": len(all_imports),
    }
    return function_code, summary


def extract_agent_timeline(trace_path: Path) -> list[dict]:
    """エージェントの思考プロセスと実行をタイムラインで抽出.
    
    OpenTelemetryトレースから、エージェントの思考プロセスと
    コード実行を時系列で抽出します。
    
    処理ロジック：
    1. 成功した execute_tool python_repl span を識別
    2. chat span から思考テキストとコード実行を抽出
    3. 成功した実行のみを含める（chat と execute_tool は同じ parent_id）
    
    Args:
        trace_path: trace.jsonlファイルのパス
    
    Returns:
        タイムラインイベントのリスト：
        [
            {
                "type": "thinking",
                "content": "エージェントの思考テキスト"
            },
            {
                "type": "code",
                "content": "実行されたPythonコード"
            }
        ]
    """
    timeline = []
    text = trace_path.read_text(encoding="utf-8")
    json_objects = []
    current_obj = ""
    brace_count = 0

    # JSON オブジェクトをパース
    for char in text:
        current_obj += char
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and current_obj.strip():
                json_objects.append(current_obj.strip())
                current_obj = ""

    # 第1パス: 成功した execute_tool python_repl の parent_id を収集
    # これらのIDは、実行が成功したchat spanを特定するのに使用される
    execute_tool_parent_ids = set()
    for json_str in json_objects:
        try:
            span = json.loads(json_str)
            if span.get("name") == "execute_tool python_repl":
                status = span.get("status", {})
                if status.get("status_code") == "OK":
                    execute_tool_parent_ids.add(span.get("parent_id"))
        except json.JSONDecodeError:
            continue

    # 第2パス: chat span から思考とコードを抽出
    # chat と execute_tool は兄弟span（同じ parent_id を共有）
    for json_str in json_objects:
        try:
            span = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        name = span.get("name")
        parent_id = span.get("parent_id")

        # chat span のみを処理
        if name != "chat":
            continue

        events = span.get("events", [])
        has_code_execution = parent_id in execute_tool_parent_ids

        for event in events:
            if event.get("name") == "gen_ai.choice":
                message_str = event.get("attributes", {}).get("message", "")
                if not message_str:
                    continue

                try:
                    message_items = json.loads(message_str)
                except json.JSONDecodeError:
                    continue

                for item in message_items:
                    # 思考テキストを抽出（toolUse なし）
                    if "text" in item and not item.get("toolUse"):
                        timeline.append({"type": "thinking", "content": item["text"]})

                    # コード実行を抽出（toolUse あり）
                    tool_use = item.get("toolUse")
                    if tool_use and tool_use.get("name") == "python_repl":
                        code = tool_use.get("input", {}).get("code", "")
                        if code:
                            # 対応する execute_tool が成功した場合のみ追加
                            if has_code_execution:
                                timeline.append({"type": "code", "content": code.strip()})
    
    return timeline


def extract_python_repl_code(trace_path: Path) -> list[str]:
    """トレースから python_repl ツール実行コードを抽出.
    
    OpenTelemetryトレースから、成功した python_repl ツール実行の
    コードブロックを抽出します。抽出対象は、対応する成功した
    execute_tool span を持つ chat span のみです。
    
    parent_id の整合性チェック：
    - chat span と execute_tool span は兄弟（同じ parent_id）
    - この整合性をチェックして、実際に実行されたコードのみを抽出
    
    Args:
        trace_path: trace.jsonlファイルのパス
    
    Returns:
        Pythonコードブロックのリスト
    """
    codes: list[str] = []
    text = trace_path.read_text(encoding="utf-8")

    json_objects = []
    current_obj = ""
    brace_count = 0

    for char in text:
        current_obj += char
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and current_obj.strip():
                json_objects.append(current_obj.strip())
                current_obj = ""

    # 第1パス: chat span と成功した execute_tool span を分類
    chat_spans = []
    execute_tool_parent_ids = set()

    for json_str in json_objects:
        try:
            span = json.loads(json_str)
            if span.get("name") == "chat":
                chat_spans.append(span)
            elif span.get("name") == "execute_tool python_repl":
                # 成功した実行の parent_id を記録
                status = span.get("status", {})
                status_code = status.get("status_code", "")
                if status_code == "OK":
                    execute_tool_parent_ids.add(span.get("parent_id"))
        except json.JSONDecodeError:
            continue

    # 第2パス: 成功した実行に対応する chat span からコードを抽出
    for chat_span in chat_spans:
        chat_parent_id = chat_span.get("parent_id")

        # 対応する成功した execute_tool がない場合はスキップ
        if chat_parent_id not in execute_tool_parent_ids:
            continue
        
        events = chat_span.get("events", [])
        for event in events:
            if event.get("name") != "gen_ai.choice":
                continue

            # メッセージ属性を取得
            attrs = event.get("attributes", {})
            message_str = attrs.get("message", "")

            if not message_str:
                continue

            # メッセージをJSON解析
            try:
                message_items = json.loads(message_str)
            except json.JSONDecodeError:
                continue

            # toolUse からコードを抽出
            for item in message_items:
                if not isinstance(item, dict):
                    continue
                tool_use = item.get("toolUse")
                if not tool_use or tool_use.get("name") != "python_repl":
                    continue

                code = tool_use.get("input", {}).get("code", "")
                if code:
                    codes.append(code.strip())

    return codes


def create_notebook(
    codes: list[str],
    output_path: Path,
    metadata: Optional[dict] = None,
    timeline: Optional[list[dict]] = None,
) -> None:
    """Jupyter ノートブックを生成.
    
    抽出されたコードとメタデータからJupyterノートブックを作成します。
    
    ノートブックの構成：
    1. タイトルと実行概要
    2. システムプロンプトとユーザーリクエスト
    3. Agentの思考プロセスとコード実行（タイムライン）
    4. 最終成果物：再利用可能な関数
    5. 開発者向けガイド
    
    Args:
        codes: 抽出されたPythonコードブロックのリスト
        output_path: 生成するノートブックのパス
        metadata: トレースメタデータ（オプション）
        timeline: タイムラインイベント（オプション）
    """
    if metadata is None:
        metadata = {}

    user_prompt = metadata.get("user_prompt", "No user prompt found in trace")
    system_prompt = metadata.get("system_prompt", "No system prompt found in trace")
    model_id = metadata.get("model_id", "Unknown model")

    nb = nbf.v4.new_notebook()

    # タイトルと概要
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "# Agent Execution Replay\n\n"
            "このノートブックは、Strands Agentが実行したPythonコードを再現可能な形で記録しています。\n\n"
            "## 実行概要\n\n"
            f"- **Model**: `{model_id}`\n"
            f"- **抽出されたコードブロック数**: {len(codes)}\n"
            f"- **トレース元**: `outputs/trace.jsonl`\n\n"
            "## トレースの仕組み\n\n"
            "Strands AgentsはOpenTelemetryを使用して実行トレースを記録します。\n"
            "トレースには以下の情報が含まれます：\n\n"
            "- LLMへのプロンプトとレスポンス\n"
            "- 使用されたツール（python_repl）とそのパラメータ\n"
            "- 実行時間とトークン使用量\n"
            "- 各サイクルの処理フロー"
        )
    )

    # システムプロンプト
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## システムプロンプト\n\n"
            "Agentに与えられた役割と責務：\n\n"
            f"```\n{system_prompt}\n```"
        )
    )
    
    # User request
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## ユーザーリクエスト\n\n"
            "Agentに与えられたタスク：\n\n"
            f"```\n{user_prompt}\n```"
        )
    )
    
    # Agent's thinking and execution timeline
    if timeline:
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                "---\n\n"
                "# Agentの思考プロセスと実行履歴\n\n"
                "以下は、Agentがタスクを実行した際の思考プロセスと実行コードを時系列で記録したものです。\n\n"
                "※このセクションは参照用です。実際の実行は「最終成果物」セクションで行ってください。"
            )
        )

        # Extract code blocks from timeline for final function building
        timeline_codes = [event["content"] for event in timeline if event["type"] == "code"]
        
        step_num = 1
        for i, event in enumerate(timeline):
            if event["type"] == "thinking":
                nb.cells.append(
                    nbf.v4.new_markdown_cell(
                        f"## Step {step_num}: Agent の思考\n\n"
                        f"{event['content']}"
                    )
                )
            elif event["type"] == "code":
                nb.cells.append(
                    nbf.v4.new_markdown_cell(
                        f"### 実行コード (Step {step_num})"
                    )
                )
                # Add code as non-executable code cell with visual styling
                code_cell = nbf.v4.new_code_cell(event["content"])
                code_cell.metadata["tags"] = ["reference-only"]
                nb.cells.append(code_cell)
                step_num += 1

        # Add final deliverable section
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                "---\n\n"
                "# 最終成果物\n\n"
                "Agentの探索フェーズで得られた知見をもとに、再利用可能な関数として実装しました。"
            )
        )
        result = build_reusable_function(timeline_codes)
        
        if result is None:
            nb.cells.append(
                nbf.v4.new_markdown_cell(
                    "## ⚠️ 関数の抽出に失敗\n\n"
                    "トレースから必須コンポーネントを抽出できませんでした。\n"
                    "以下が不足している可能性があります:\n"
                    "- `filter_aws_articles` 関数ブロック\n"
                    "- 必要なimport文\n"
                    "- マーカーコメント (`# === FINAL_FUNCTION_START/END ===`)"
                )
            )
        else:
            function_code, summary = result
            
            # Step 1: Function definition
            extraction_mode = summary.get("extraction_mode", "unknown")
            if extraction_mode == "marker":
                extraction_info = (
                    f"- 抽出方法: マーカーベース (`# === FINAL_FUNCTION_START/END ===`)\n"
                    f"- ソースコードブロック数: {summary['source_blocks']}\n"
                    f"- 抽出元ブロック: {summary['source_block_index']}\n"
                    f"- import文数: {summary['imports']}"
                )
            elif extraction_mode == "block":
                extraction_info = (
                    f"- 抽出方法: 関数ブロック検出\n"
                    f"- ソースコードブロック数: {summary['source_blocks']}\n"
                    f"- 関数名: `{summary['function_name']}`\n"
                    f"- 抽出元ブロック: {summary['source_block_index']}\n"
                    f"- import文数: {summary['imports']}"
                )
            else:
                extraction_info = (
                    f"- 抽出方法: AST再構成\n"
                    f"- ソースコードブロック数: {summary['source_blocks']}\n"
                    f"- import文数: {summary['imports']}"
                )
            
            nb.cells.append(
                nbf.v4.new_markdown_cell(
                    "## 1. 関数定義\n\n"
                    "トレースから抽出した再利用可能な関数です。\n\n"
                    f"{extraction_info}"
                )
            )
            
            # Extract and display implementation rationale
            rationale = _extract_implementation_rationale(timeline_codes)
            if rationale:
                nb.cells.append(
                    nbf.v4.new_markdown_cell(
                        "### 実装の設計判断と根拠\n\n"
                        f"{rationale}"
                    )
                )
            
            nb.cells.append(nbf.v4.new_code_cell(function_code))
            
            # Step 2: Function execution (extract from FUNCTION_CALL marker)
            function_call = _extract_marked_block(
                timeline_codes,
                "# === FUNCTION_CALL_START ===",
                "# === FUNCTION_CALL_END ===",
            )
            
            nb.cells.append(
                nbf.v4.new_markdown_cell(
                    "## 2. 関数の実行\n\n"
                    "パラメータを編集して関数を実行します。このセルを編集することで、実行時の値を調整できます。"
                )
            )
            
            if function_call:
                execution_code, _ = function_call
                nb.cells.append(nbf.v4.new_code_cell(execution_code))
            else:
                # Fallback if FUNCTION_CALL marker not found
                nb.cells.append(
                    nbf.v4.new_code_cell(
                        "# 関数を実行（適切なパラメータを指定してください）\n"
                        "# result = your_function(param1=value1, param2=value2)\n"
                        "# print(result)"
                    )
                )
    
    elif not codes:
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                "## ⚠️ 実行コード\n\n"
                "トレースファイルから `python_repl` のコードが見つかりませんでした。"
            )
        )
    else:
        # Fallback: single combined code block
        final_code = codes[-1] if len(codes) == 1 else "\n\n".join(codes)
        
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                f"## 実行コード\n\n"
                f"以下は、Agentが `python_repl` ツールで実行した実際のコードです。\n"
                f"セルを実行すると、Agentと同じ処理を再現できます。\n\n"
                f"**コードブロック数**: {len(codes)}"
            )
        )
        
        nb.cells.append(nbf.v4.new_code_cell(final_code))
    
    # Add guidance for executing runnable cells
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "---\n\n"
            "## ✅ 実行ガイド\n\n"
            "実行可能なセルを上から順番に実行すると、エージェントがユーザーからのリクエストを解決するために作成した関数を試行できます。\n\n"
            "1. **関数定義**: セル1を実行して関数を定義\n"
            "2. **関数実行**: セル2のパラメータを編集してから実行\n"
        )
    )
    
    # How to use section
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "---\n\n"
            "# 開発者向けガイド\n\n"
            "## Agentの再実行\n\n"
            "ソースコードを変更した場合は、以下のコマンドでAgentを再実行してください：\n\n"
            "```bash\n"
            "# トレース付きで実行\n"
            "BYPASS_TOOL_CONSENT=true uv run python src/main.py\n"
            "```\n\n"
            "実行すると以下が自動生成されます：\n"
            "- `outputs/trace.jsonl` - 完全な実行トレース\n"
            "- `outputs/agent_replay.ipynb` - このノートブック\n\n"
            "### トレースファイルの確認\n\n"
            "`outputs/trace.jsonl` には完全な実行トレースがJSON形式で保存されています。\n"
            "各オブジェクトは OpenTelemetry の Span を表し、以下の情報が含まれます：\n\n"
            "- `gen_ai.user.message`: ユーザーからのプロンプト\n"
            "- `gen_ai.choice`: LLMの応答（テキストまたはツール使用）\n"
            "- `gen_ai.tool.name`: 使用されたツール名\n"
            "- `gen_ai.usage.*`: トークン使用量\n"
            "- `gen_ai.event.start_time/end_time`: 実行時刻"
        )
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def main() -> None:
    """トレースからノートブックを生成.
    
    このメイン関数は以下の処理を実行します：
    1. trace.jsonl からメタデータを抽出
    2. 実行されたPythonコードを抽出
    3. エージェントの思考プロセスをタイムラインとして抽出
    4. 上記から Jupyter ノートブックを生成
    
    出力：
    - outputs/agent_replay.ipynb: 対話的なノートブック
    """
    trace_path = Path("outputs/trace.jsonl")
    output_path = Path("outputs/agent_replay.ipynb")

    if not trace_path.exists():
        print(f"❌ Trace file not found: {trace_path}")
        print("   Run the agent first: BYPASS_TOOL_CONSENT=true uv run python src/main.py")
        return

    metadata = extract_trace_metadata(trace_path)
    codes = extract_python_repl_code(trace_path)
    timeline = extract_agent_timeline(trace_path)
    create_notebook(codes, output_path, metadata, timeline)
    print(f"✅ Notebook created: {output_path}")
    print(f"   Extracted {len(codes)} code block(s) from trace")
    print(f"   Timeline events: {len(timeline)}")


if __name__ == "__main__":
    main()

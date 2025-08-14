from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel


designer = Agent(
    model=BedrockModel(
        model_id="us.amazon.nova-micro-v1:0",
        region_name="us-east-1",
    ),
    system_prompt=(
        """
        あなたはプロのプログラム仕様のデザイナーです。

        # 作成物定義
        最終的に出力するプログラム仕様は以下の構成と内容とする
        | セクション名 | 内容 |
        | --- | --- |
        | 概要 | プログラムの目的、期待される結果 |
        | 入力 | 入力値の仕様、バリエーション、範囲、型 |
        | オプション | 指定可能なオプション、指定による挙動の変化、必須or任意 |
        | 入力 | 正常応答の仕様、バリエーション、範囲、型 |
        | エラー | 異常応答の仕様、バリエーション、範囲、型 |

        # 作業ステップ
        1. ユーザーの要望から推定されるプログラム仕様を作成する
        2. 完成した仕様を返却する

        # 作業ルール
        - 出力は仕様のみ
        - 解説や経緯など仕様以外は出力しない
        - ユーザーからの指定がない項目は自由に設計してよい
        - 曖昧さを排除し、明確な記述とする

        # 出力フォーマット
        - markdown形式
        """
    )
)
coder = Agent(
    model=BedrockModel(
        model_id="us.amazon.nova-micro-v1:0",
        region_name="us-east-1",
    ),
    system_prompt=(
        """
        あなたはプロのプログラマーです。
        ユーザーから提示された仕様に基づいてプログラムを作成します。

        # 作成物定義
        最終的に出力するプログラムは以下の内容とする
        - ソースコードファイルにそのまま貼り付け可能な内容とする
        - 解説や実行方法、サンプル、装飾行なども一切出力しない
        - 実装コメントと関数やモジュールに対するドキュメンテーションコメントは含める。
        - 適切にインデントを整え、フォーマットする。

        # 作業ステップ
        1. プログラムを記述する
        2. 実装コメントを追加する
        3. ドキュメンテーションコメントを追加する
        4. フォーマットする

        # 作業ルール
        - 出力はプログラムのみ
        - 解説や経緯などプログラム以外は出力しない

        # 出力フォーマット
        - 利用したプログラム言語のソースコード
        """
    )
)

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    """Process user input and return a response"""
    requirements = payload.get("message")

    try:
        print("\n===== ===== =====")
        specifications = designer(requirements)
        print("\n===== ===== =====")
        sources = coder(str(specifications))
        print("\n===== ===== =====")
        
        return {"result": str(sources) + "\n"}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run()
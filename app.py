from pathlib import Path
import base64
import textwrap

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from cat_images import display_cat_images

# =========================
# 基本設定
# =========================

DATABASE_DIRECTORY = "chroma_db"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

load_dotenv()

# =========================
# CSS読み込み
# =========================

def load_css(css_path: str) -> None:

    with open(
        css_path,
        "r",
        encoding="utf-8"
    ) as css_file:

        css = css_file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


# =========================
# HTML表示
# =========================

def show_html(html: str) -> None:

    st.html(
        textwrap.dedent(html)
    )


# =========================
# 画像をBase64に変換
# =========================

def get_base64_image(
    image_path: str
) -> str:

    with open(
        image_path,
        "rb"
    ) as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode()

# =========================
# Chroma読み込み
# =========================

@st.cache_resource
def load_vectorstore() -> Chroma:

    database_path = Path(
        DATABASE_DIRECTORY
    )

    if not database_path.exists():

        raise FileNotFoundError(
            "chroma_dbが見つかりません。"
            "先にcreate_db.pyを実行してください。"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = Chroma(
        persist_directory=DATABASE_DIRECTORY,
        embedding_function=embeddings
    )

    return vectorstore


# =========================
# AI読み込み
# =========================

@st.cache_resource
def load_llm() -> ChatGoogleGenerativeAI:

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )


# =========================
# Query Expansion
# 検索語を意味的に拡張
# =========================

def expand_query(
    llm: ChatGoogleGenerativeAI,
    question: str
) -> list:

    """
    ユーザーの質問と同じ意味を持つ
    検索表現を最大4件生成する。

    例:
    くろねこの特徴
    ↓
    黒猫の特徴
    クロネコの特徴
    黒い猫の特徴
    """

    prompt = f"""
あなたは日本語検索のQuery Expansionを行うAIです。

次の質問について、
意味を変えずに検索で使える別表現を作ってください。

特に以下を考慮してください。

・ひらがな、カタカナ、漢字の表記違い
・同じ意味を持つ自然な日本語表現
・猫種名、毛色、模様などの一般的な表記
・質問の意味を勝手に広げない
・新しい情報を追加しない
・最大4件まで
・説明は書かない
・1行に1つだけ書く

【質問】
{question}
"""

    try:

        response = llm.invoke(
            prompt
        )

        content = response.content

        # Geminiがリスト形式で返した場合
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        text_parts.append(text)

                elif isinstance(item, str):

                    text_parts.append(item)

            generated_text = "\n".join(
                text_parts
            )

        else:

            generated_text = str(
                content
            )

        generated_lines = (
            generated_text
            .strip()
            .splitlines()
        )

        expanded_queries = []

        # 元の質問は必ず残す
        expanded_queries.append(
            question.strip()
        )

        for line in generated_lines:

            cleaned = (
                line
                .strip()
                .lstrip("・")
                .lstrip("-")
                .strip()
            )

            # 「1.」などを除く
            if (
                len(cleaned) >= 2
                and cleaned[0].isdigit()
                and cleaned[1] in [".", "．", "、", ")"]
            ):
                cleaned = cleaned[2:].strip()

            if (
                cleaned
                and cleaned not in expanded_queries
            ):

                expanded_queries.append(
                    cleaned
                )

            if len(expanded_queries) >= 5:
                break

        return expanded_queries

    except Exception:

        # Query Expansionに失敗しても
        # 元の質問だけで検索できる
        return [
            question.strip()
        ]


# =========================
# 不要表現を除く
# =========================

def clean_question(
    question: str
) -> str:

    return (
        question
        .replace("について", "")
        .replace("教えてください", "")
        .replace("教えて", "")
        .replace("とは", "")
        .replace("って", "")
        .replace("？", "")
        .replace("?", "")
        .strip()
    )


# =========================
# 資料検索
# =========================

def search_documents(
    vectorstore: Chroma,
    llm: ChatGoogleGenerativeAI,
    question: str,
    search_count: int = 5
) -> tuple[list, list]:

    """
    1. Query Expansion
    2. 各検索表現でSimilarity Search
    3. 候補を統合
    4. キーワード一致も加点
    5. 上位5件を返す
    """

    # -------------------------
    # Query Expansion
    # -------------------------

    expanded_queries = expand_query(
        llm,
        question
    )

    scored_documents = {}

    # -------------------------
    # 複数の検索語で
    # Similarity Search
    # -------------------------

    for query_number, query in enumerate(
        expanded_queries
    ):

        candidates = (
            vectorstore.similarity_search(
                query,
                k=30
            )
        )

        cleaned_query = clean_question(
            query
        )

        for index, document in enumerate(
            candidates
        ):

            file_name = (
                document.metadata.get(
                    "source_file",
                    ""
                )
            )

            page_number = (
                document.metadata.get(
                    "page"
                )
            )

            content = (
                document.page_content
                .strip()
            )

            key = (
                file_name,
                page_number,
                content
            )

            # Chroma順位による基本点
            score = 30 - index

            # 元質問を少し優先
            if query_number == 0:
                score += 5

            # 検索表現全体が一致
            if (
                cleaned_query
                and cleaned_query in content
            ):
                score += 50

            # 2〜8文字の部分一致
            for length in range(
                min(
                    8,
                    len(cleaned_query)
                ),
                1,
                -1
            ):

                for start in range(
                    len(cleaned_query)
                    - length
                    + 1
                ):

                    keyword = (
                        cleaned_query[
                            start:
                            start + length
                        ]
                    )

                    if keyword in content:
                        score += length

            # 同じチャンクが
            # 別検索語でも見つかった場合
            # その結果も少し評価する
            if key in scored_documents:

                old_score = (
                    scored_documents[
                        key
                    ]["score"]
                )

                scored_documents[
                    key
                ]["score"] = (
                    max(
                        old_score,
                        score
                    )
                    + 3
                )

            else:

                scored_documents[key] = {
                    "score": score,
                    "document": document
                }

    # -------------------------
    # 点数順に並べる
    # -------------------------

    ranked_results = sorted(
        scored_documents.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    final_results = [
        item["document"]
        for item in ranked_results[
            :search_count
        ]
    ]

    return (
        final_results,
        expanded_queries
    )


# =========================
# AIへ渡す参考資料
# =========================

def create_context(
    results: list
) -> str:

    context_parts = []

    for document in results:

        file_name = (
            document.metadata.get(
                "source_file",
                "ファイル名不明"
            )
        )

        page_number = (
            document.metadata.get(
                "page"
            )
        )

        if page_number is not None:

            page_text = (
                f"{page_number + 1}ページ"
            )

        else:

            page_text = "ページ不明"

        context_part = f"""
【ファイル名】
{file_name}

【ページ】
{page_text}

【内容】
{document.page_content}
"""

        context_parts.append(
            context_part
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# =========================
# AI回答生成
# =========================

def generate_answer(
    llm: ChatGoogleGenerativeAI,
    question: str,
    context: str,
    mode: str
) -> str:

    if mode == "猫図鑑検索":

        system_instruction = """
あなたは猫図鑑AIです。

登録されている猫の資料をもとに、
猫種、毛色、模様、耳、しっぽ、
体の特徴などについて回答してください。

回答方針:
・質問への答えを最初に分かりやすく示してください
・難しい言葉はできるだけ避けてください
・猫種と毛色や模様を混同しないでください
・資料にある特徴を整理して説明してください
・資料にない情報は推測しないでください

回答形式:

答え:
質問への直接的な回答

特徴:
・特徴1
・特徴2
・特徴3

補足:
必要な場合だけ補足説明

参考資料:
・ファイル名（ページ番号）
"""

    elif mode == "猫との暮らし相談":

        system_instruction = """
あなたは猫との暮らしをサポートするAIです。

登録されている猫の資料をもとに、
猫との暮らし、日々のお世話、
猫の行動などについて
分かりやすく回答してください。

回答方針:
・飼い主に分かりやすい言葉で説明してください
・質問に関係する猫の行動の理由を説明してください
・資料に対応方法があれば具体的に紹介してください
・猫の性格には個体差があることを考慮してください
・資料にない情報は推測しないでください
・医療的な診断は行わないでください

回答形式:

ご案内:
質問への分かりやすい回答

考えられる理由:
資料から分かる猫の行動や特徴

対応方法:
1. 対応方法
2. 対応方法

参考資料:
・ファイル名（ページ番号）
"""

    else:

        system_instruction = """
あなたは猫種を比較する猫図鑑AIです。

登録されている猫種資料をもとに、
複数の猫種の特徴を比較してください。

回答方針:
・猫種ごとの違いが分かるように整理してください
・原産地、体格、毛の長さ、見た目、
  性格の傾向などを比較してください
・性格はあくまで傾向であり、
  個体差があることを明記してください
・資料にない項目は
  「資料からは確認できません」
  としてください
・見た目だけで猫種を断定しないでください

回答形式:

比較する猫:
質問に含まれる猫種

主な違い:
猫種ごとの特徴を分かりやすく比較

共通点:
資料から確認できる共通点

選ぶときのポイント:
資料から分かる範囲で整理

参考資料:
・ファイル名（ページ番号）
"""

    prompt = f"""
{system_instruction}

【参考資料】
{context}

【質問】
{question}

【共通ルール】
・必ず参考資料だけを利用してください
・参考資料に書かれていない内容は推測しないでください
・一般的な知識を勝手に追加しないでください
・資料から回答できない場合は
「資料からは確認できません」
と回答してください
・最後に必ず参考ファイル名とページ番号を示してください
・見出しの後は必ず改行してください
・箇条書きの「・」は必ず行の先頭に置いてください
・箇条書きは1項目ごとに必ず改行してください
・複数の箇条書きを同じ行に書かないでください
"""

    response = llm.invoke(
        prompt
    )

    content = response.content

    # =========================
    # Geminiの回答を文字列にする
    # =========================

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get(
                    "text",
                    ""
                )

                if text:
                    text_parts.append(text)

            elif isinstance(item, str):

                text_parts.append(item)

        answer_text = "\n".join(
            text_parts
        )

    else:

        answer_text = str(
            content
        )


    # =========================
    # 表示用に改行を整理
    # =========================

    # 「 ・」を箇条書きごとの改行にする
    answer_text = answer_text.replace(
        " ・",
        "\n・"
    )

    # Geminiがすでに作った余分な空行を削除
    while "\n\n" in answer_text:
        answer_text = answer_text.replace(
            "\n\n",
            "\n"
        )

    # 見出しの後を1回だけ改行
    headings = [
        "答え:",
        "特徴:",
        "補足:",
        "参考資料:",
        "ご案内:",
        "考えられる理由:",
        "対応方法:",
        "比較する猫:",
        "主な違い:",
        "共通点:",
        "選ぶときのポイント:"
    ]

    for heading in headings:
        answer_text = answer_text.replace(
            heading,
            heading + "\n"
        )

    # 最後にもう一度、余分な空行を削除
    while "\n\n" in answer_text:
        answer_text = answer_text.replace(
            "\n\n",
            "\n"
        )

    return answer_text.strip()


# =========================
# 検索語表示
# =========================

def display_expanded_queries(
    queries: list
) -> None:

    with st.expander(
        "🔎 検索に使った表現を見る"
    ):

        for index, query in enumerate(
            queries,
            start=1
        ):

            st.write(
                f"{index}. {query}"
            )


# =========================
# 検索された資料表示
# =========================

def display_sources(
    results: list
) -> None:

    with st.expander(
        "📚 検索された参考資料を見る"
    ):

        for index, document in enumerate(
            results,
            start=1
        ):

            file_name = (
                document.metadata.get(
                    "source_file",
                    "ファイル名不明"
                )
            )

            page_number = (
                document.metadata.get(
                    "page"
                )
            )

            if page_number is not None:

                page_text = (
                    f"{page_number + 1}ページ"
                )

            else:

                page_text = "ページ不明"

            st.markdown(
                f"### 🐾 検索結果 {index}"
            )

            st.write(
                f"ファイル：{file_name}"
            )

            st.write(
                f"ページ：{page_text}"
            )

            st.write(
                document.page_content
            )

            st.divider()


# =========================
# メイン画面
# =========================

def main():

    st.set_page_config(
        page_title="CAT RAG｜猫図鑑AI",
        page_icon="🐱",
        layout="wide"
    )

    load_css(
        "style.css"
    )

    # =========================
    # 背景画像
    # =========================

    background_image = (
        get_base64_image(
            "images/background.png"
        )
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.12),
                    rgba(255, 255, 255, 0.12)
                ),
                url(
                    "data:image/png;base64,{background_image}"
                );
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # 左メニュー
    # =========================

    st.sidebar.markdown(
        """
        ## 🐾 CAT RAG
        ### 猫図鑑AI
        """
    )

    mode = st.sidebar.selectbox(
        "モードを選択してください",
        [
            "猫図鑑検索",
            "猫との暮らし相談",
            "猫種比較"
        ]
    )

    # =========================
    # モード設定
    # =========================

    if mode == "猫図鑑検索":

        title = "猫図鑑AI"

        description = (
            "猫種・毛色・模様・耳・しっぽなど、"
            "猫の特徴について質問できます。"
        )

        placeholder = (
            "例：はちわれって猫の種類なの？"
        )

    elif mode == "猫との暮らし相談":

        title = "猫との暮らし相談"

        description = (
            "猫のお世話や行動について、"
            "登録された資料をもとに回答します。"
        )

        placeholder = (
            "例：猫が爪とぎをするのはなぜ？"
        )

    else:

        title = "猫種比較"

        description = (
            "複数の猫種の特徴を、"
            "登録された猫図鑑をもとに比較します。"
        )

        placeholder = (
            "例：ラグドールと"
            "メインクーンの違いは？"
        )

    # =========================
    # 上部
    # =========================

    header_left, header_right = (
        st.columns(
            [1.1, 2.3],
            vertical_alignment="center"
        )
    )

    with header_left:

        st.image(
            "images/cat_main.png",
            width=280
        )

    with header_right:

        show_html(
            f"""
            <div class="main-header">

                <div class="main-logo">
                    CAT RAG 🐾
                </div>

                <div class="main-title">
                    {title}
                </div>

                <div class="main-description">
                    猫に関するPDF資料の内容について
                    質問できます。
                </div>

            </div>
            """
        )

    # =========================
    # 検索エリア
    # =========================

    show_html(
        """
        <div class="search-heading">
            🐾 猫について調べてみよう
        </div>
        """
    )

    st.caption(
        description
    )

    # =========================
    # Chroma・OpenAI
    # =========================

    try:

        vectorstore = (
            load_vectorstore()
        )

        llm = (
            load_llm()
        )

    except Exception as error:

        st.error(
            str(error)
        )

        st.stop()

    # =========================
    # 質問入力
    # =========================

    search_left, search_right = (
        st.columns(
            [5, 1.3],
            vertical_alignment="bottom"
        )
    )

    with search_left:

        question = st.text_input(
            "質問",
            placeholder=placeholder,
            label_visibility="collapsed"
        )

    with search_right:

        search_button = st.button(
            "🔍 検索する",
            type="primary",
            use_container_width=True
        )

    # =========================
    # 質問例
    # =========================

    show_html(
        """
        <div class="example-heading">
            こんな質問ができます
        </div>
        """
    )

    (
        example1,
        example2,
        example3,
        example4
    ) = st.columns(4)

    with example1:

        show_html(
            """
            <div class="example-chip">
                🐾 はちわれって猫の種類？
            </div>
            """
        )

    with example2:

        show_html(
            """
            <div class="example-chip">
                🐾 くろねこの特徴を教えて
            </div>
            """
        )

    with example3:

        show_html(
            """
            <div class="example-chip">
                🐾 ラグドールの性格は？
            </div>
            """
        )

    with example4:

        show_html(
            """
            <div class="example-chip">
                🐾 猫のしっぽの役割は？
            </div>
            """
        )

    # =========================
    # 検索
    # =========================

    if search_button:

        if not question.strip():

            st.warning(
                "質問を入力してください。"
            )

            st.stop()

        with st.spinner(
            "🐱 猫の資料を検索しています..."
        ):

            (
                results,
                expanded_queries
            ) = search_documents(
                vectorstore=vectorstore,
                llm=llm,
                question=question,
                search_count=5
            )

            if not results:

                st.warning(
                    "関連する資料が"
                    "見つかりませんでした。"
                )

                st.stop()

            context = (
                create_context(
                    results
                )
            )

            answer = (
                generate_answer(
                    llm=llm,
                    question=question,
                    context=context,
                    mode=mode
                )
            )
        
        # =========================
        # 猫画像表示
        # =========================

        display_cat_images(
            question,
            expanded_queries
        )

        # =========================
        # AI回答
        # =========================

        show_html(
            """
            <div class="answer-title">
                🐈 AI回答
            </div>
            """
        )

        with st.container(
            border=True
        ):

            formatted_answer = answer.replace(
                "\n",
                "<br>"
            )

            st.markdown(
                formatted_answer,
                unsafe_allow_html=True
            )

        # =========================
        # 検索に使った表現
        # =========================

        display_expanded_queries(
            expanded_queries
        )

        # =========================
        # 参考資料
        # =========================

        display_sources(
            results
        )


# =========================
# 起動
# =========================

if __name__ == "__main__":
    main()
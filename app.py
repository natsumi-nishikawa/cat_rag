from pathlib import Path
import base64
import textwrap

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# =========================
# 基本設定
# =========================

DATABASE_DIRECTORY = "chroma_db"

CAT_IMAGE_DIRECTORY = Path(
    "images/cats"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

load_dotenv()


# =========================
# 猫画像設定
# =========================

CAT_IMAGE_DEFINITIONS = [
    {
        "name": "マンチカン",
        "file": "munchkin.png",
        "keywords": [
            "マンチカン",
            "munchkin"
        ]
    },
    {
        "name": "スコティッシュフォールド",
        "file": "scottish_fold.png",
        "keywords": [
            "スコティッシュフォールド",
            "スコティッシュ",
            "scottishfold"
        ]
    },
    {
        "name": "ラグドール",
        "file": "ragdoll.png",
        "keywords": [
            "ラグドール",
            "ragdoll"
        ]
    },
    {
        "name": "メインクーン",
        "file": "maine_coon.png",
        "keywords": [
            "メインクーン",
            "mainecoon"
        ]
    },
    {
        "name": "ロシアンブルー",
        "file": "russian_blue.png",
        "keywords": [
            "ロシアンブルー",
            "russianblue"
        ]
    },
    {
        "name": "ノルウェージャンフォレストキャット",
        "file": "norwegian_forest.png",
        "keywords": [
            "ノルウェージャンフォレストキャット",
            "ノルウェージャン",
            "norwegianforestcat"
        ]
    },
    {
        "name": "ベンガル",
        "file": "bengal.png",
        "keywords": [
            "ベンガル",
            "bengal"
        ]
    },
    {
        "name": "シャム",
        "file": "siamese.png",
        "keywords": [
            "シャム",
            "シャム猫",
            "siamese"
        ]
    },
    {
        "name": "ブリティッシュショートヘア",
        "file": "british_shorthair.png",
        "keywords": [
            "ブリティッシュショートヘア",
            "ブリティッシュ",
            "britishshorthair"
        ]
    },
    {
        "name": "アビシニアン",
        "file": "abyssinian.png",
        "keywords": [
            "アビシニアン",
            "abyssinian"
        ]
    },
    {
        "name": "白猫",
        "file": "white_cat.png",
        "keywords": [
            "白猫",
            "白い猫",
            "whitecat"
        ]
    },
    {
        "name": "黒猫",
        "file": "black_cat.png",
        "keywords": [
            "黒猫",
            "黒い猫",
            "blackcat"
        ]
    },
    {
        "name": "ハチワレ",
        "file": "hachiware.png",
        "keywords": [
            "ハチワレ",
            "はちわれ",
            "八割れ",
            "tuxedocat"
        ]
    },
    {
        "name": "三毛猫",
        "file": "calico.png",
        "keywords": [
            "三毛猫",
            "三毛",
            "みけねこ",
            "calicocat"
        ]
    },
    {
        "name": "サビ猫",
        "file": "tortoiseshell.png",
        "keywords": [
            "サビ猫",
            "さび猫",
            "サビ",
            "さびねこ",
            "tortoiseshellcat"
        ]
    }
]


# =========================
# CSS読み込み
# =========================

def load_css(css_path: str) -> None:
    """
    外部CSSファイルを読み込む。
    """

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
# 質問から猫を判定
# =========================

def normalize_text(
    text: str
) -> str:
    """
    猫名を判定しやすい形に整える。
    """

    return (
        text
        .lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("-", "")
        .replace("_", "")
    )


def find_cat_images(
    question: str
) -> list:
    """
    質問文に含まれる猫種・毛色を判定し、
    表示対象の画像情報を返す。
    """

    normalized_question = normalize_text(
        question
    )

    matched_cats = []

    for cat in CAT_IMAGE_DEFINITIONS:

        for keyword in cat["keywords"]:

            normalized_keyword = normalize_text(
                keyword
            )

            if (
                normalized_keyword
                in normalized_question
            ):

                matched_cats.append(
                    cat
                )

                break

    return matched_cats


# =========================
# 猫画像を表示
# =========================

def display_cat_images(
    question: str
) -> None:
    """
    質問に含まれている猫の画像を表示する。
    """

    matched_cats = find_cat_images(
        question
    )

    if not matched_cats:
        return

    show_html(
        """
        <div class="answer-title">
            🐱 質問に登場した猫
        </div>
        """
    )

    # 1匹の場合
    if len(matched_cats) == 1:

        cat = matched_cats[0]

        image_path = (
            CAT_IMAGE_DIRECTORY
            / cat["file"]
        )

        if image_path.exists():

            st.image(
                str(image_path),
                caption=cat["name"],
                width=650
            )

        else:

            st.warning(
                f"{cat['name']}の画像が"
                f"見つかりません："
                f"{image_path}"
            )

    # 2匹以上の場合
    else:

        columns = st.columns(
            min(
                len(matched_cats),
                3
            )
        )

        for index, cat in enumerate(
            matched_cats
        ):

            image_path = (
                CAT_IMAGE_DIRECTORY
                / cat["file"]
            )

            column = columns[
                index
                % len(columns)
            ]

            with column:

                if image_path.exists():

                    st.image(
                        str(image_path),
                        caption=cat["name"],
                        use_container_width=True
                    )

                else:

                    st.warning(
                        f"{cat['name']}の"
                        "画像がありません"
                    )


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
            "先にcreate_db.pyを"
            "実行してください。"
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
# OpenAI読み込み
# =========================

@st.cache_resource
def load_llm() -> ChatOpenAI:

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    return llm


# =========================
# 資料検索
# =========================

def search_documents(
    vectorstore: Chroma,
    question: str,
    search_count: int = 5
) -> list:

    """
    Chromaから候補を広めに検索し、
    質問に含まれる言葉も考慮して
    上位5件を返す。
    """

    # Chromaから30件取得
    candidates = (
        vectorstore.similarity_search(
            question,
            k=30
        )
    )

    # 質問から不要な表現を除く
    cleaned_question = (
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

    scored_results = []

    for index, document in enumerate(
        candidates
    ):

        content = (
            document.page_content.strip()
        )

        # Chroma順位を基本点にする
        score = 30 - index

        # 質問全体の一致
        if (
            cleaned_question
            and cleaned_question in content
        ):
            score += 50

        # 2〜8文字の部分一致
        for length in range(
            min(
                8,
                len(cleaned_question)
            ),
            1,
            -1
        ):

            for start in range(
                len(cleaned_question)
                - length
                + 1
            ):

                keyword = (
                    cleaned_question[
                        start:
                        start + length
                    ]
                )

                if keyword in content:
                    score += length

        scored_results.append(
            (
                score,
                document
            )
        )

    # 点数が高い順
    scored_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    unique_results = []

    seen = set()

    for score, document in scored_results:

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
            document.page_content.strip()
        )

        key = (
            file_name,
            page_number,
            content
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_results.append(
            document
        )

        if (
            len(unique_results)
            >= search_count
        ):
            break

    return unique_results


# =========================
# AIへ渡す参考資料を作成
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

            page_text = (
                "ページ不明"
            )

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
    llm: ChatOpenAI,
    question: str,
    context: str,
    mode: str
) -> str:

    # -------------------------
    # 猫図鑑検索
    # -------------------------

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

    # -------------------------
    # 猫との暮らし相談
    # -------------------------

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

    # -------------------------
    # 猫種比較
    # -------------------------

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
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# =========================
# 検索された資料を表示
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

                page_text = (
                    "ページ不明"
                )

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

    # =========================
    # ページ設定
    # =========================

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
            "例：ハチワレって猫の種類なの？"
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
                🐾 ハチワレって猫の種類？
            </div>
            """
        )

    with example2:

        show_html(
            """
            <div class="example-chip">
                🐾 三毛猫の特徴を教えて
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

            results = (
                search_documents(
                    vectorstore=vectorstore,
                    question=question,
                    search_count=5
                )
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
        # 猫画像
        # =========================

        display_cat_images(
            question
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

            st.markdown(
                answer
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
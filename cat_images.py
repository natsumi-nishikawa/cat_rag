from pathlib import Path

import streamlit as st


# =========================
# 猫画像フォルダ
# =========================

CAT_IMAGE_DIRECTORY = Path("images/cats")


# =========================
# 猫画像設定
# =========================

CAT_IMAGE_DEFINITIONS = [
    {
        "name": "アビシニアン",
        "file": "abyssinian.png",
        "keywords": ["アビシニアン", "abyssinian"]
    },
    {
        "name": "ベンガル",
        "file": "bengal.png",
        "keywords": ["ベンガル", "bengal"]
    },
    {
        "name": "黒猫",
        "file": "black_cat.png",
        "keywords": [
            "黒猫", "黒ねこ", "くろねこ",
            "くろ猫", "クロネコ", "クロ猫",
            "黒い猫", "blackcat"
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
        "name": "三毛猫",
        "file": "calico.png",
        "keywords": [
            "三毛猫", "三毛", "みけねこ",
            "みけ猫", "ミケネコ", "calico"
        ]
    },
    {
        "name": "ハチワレ",
        "file": "hachiware.png",
        "keywords": [
            "ハチワレ", "はちわれ",
            "八割れ", "八われ", "hachiware"
        ]
    },
    {
        "name": "メインクーン",
        "file": "maine_coon.png",
        "keywords": ["メインクーン", "mainecoon"]
    },
    {
        "name": "マンチカン",
        "file": "munchikin.png",
        "keywords": ["マンチカン", "munchkin"]
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
        "name": "ラグドール",
        "file": "ragdoll.png",
        "keywords": ["ラグドール", "ragdoll"]
    },
    {
        "name": "ロシアンブルー",
        "file": "russian_blue.png",
        "keywords": ["ロシアンブルー", "russianblue"]
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
        "name": "シャム",
        "file": "siamese.png",
        "keywords": [
            "シャム", "シャム猫",
            "siamese"
        ]
    },
    {
        "name": "サビ猫",
        "file": "tortoiseshell.png",
        "keywords": [
            "サビ猫", "さび猫",
            "サビねこ", "さびねこ",
            "サビ", "tortoiseshell"
        ]
    },
    {
        "name": "白猫",
        "file": "white_cat.png",
        "keywords": [
            "白猫", "白ねこ", "しろねこ",
            "しろ猫", "シロネコ", "シロ猫",
            "白い猫", "whitecat"
        ]
    }
]


# =========================
# 文字列整理
# =========================

def normalize_text(
    text: str
) -> str:

    return (
        text
        .lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("-", "")
        .replace("_", "")
    )


# =========================
# 質問から猫画像を判定
# =========================

def find_cat_images(
    question: str,
    expanded_queries: list | None = None
) -> list:

    texts = [question]

    if expanded_queries:
        texts.extend(
            expanded_queries
        )

    combined_text = normalize_text(
        " ".join(texts)
    )

    matched_cats = []

    for cat in CAT_IMAGE_DEFINITIONS:

        for keyword in cat["keywords"]:

            normalized_keyword = normalize_text(
                keyword
            )

            if normalized_keyword in combined_text:

                matched_cats.append(
                    cat
                )

                break

    return matched_cats


# =========================
# 猫画像表示
# =========================

def display_cat_images(
    question: str,
    expanded_queries: list | None = None
) -> None:

    matched_cats = find_cat_images(
        question,
        expanded_queries
    )

    if not matched_cats:
        return

    st.markdown(
        "### 🐱 質問に関連する猫"
    )

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
                f"画像が見つかりません：{cat['file']}"
            )

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

            with columns[
                index % len(columns)
            ]:

                if image_path.exists():

                    st.image(
                        str(image_path),
                        caption=cat["name"],
                        use_container_width=True
                    )

                else:

                    st.warning(
                        f"画像が見つかりません：{cat['file']}"
                    )
from pathlib import Path
import shutil
import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# 基本設定
# =========================

PDF_DIRECTORY = "sample_pdf"

DATABASE_DIRECTORY = "chroma_db"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# PDF文章の空白を整理
# =========================

def clean_pdf_text(text: str) -> str:
    """
    PDFから抽出された文章に入っている
    不要な空白をできるだけ取り除く。
    """

    # 改行を一度スペースにする
    text = text.replace("\n", " ")

    # 全角スペースも半角スペースにする
    text = text.replace("　", " ")

    # PDFで1文字ずつ分離された日本語を結合する
    # 例：
    # ラ グ ドー ル → ラグドール
    # 猫 の し っ ぽ → 猫のしっぽ
    text = re.sub(
        r'(?<=[ぁ-んァ-ヶ一-龥々ー])'
        r' +'
        r'(?=[ぁ-んァ-ヶ一-龥々ー])',
        '',
        text
    )

    # 日本語と句読点の間
    text = re.sub(
        r' +([、。！？])',
        r'\1',
        text
    )

    # 記号の後ろ
    text = re.sub(
        r'([「（【]) +',
        r'\1',
        text
    )

    # 記号の前
    text = re.sub(
        r' +([」）】])',
        r'\1',
        text
    )

    # 最後に残った連続スペースを1個にする
    text = re.sub(
        r' +',
        ' ',
        text
    )

    return text.strip()

# =========================
# 猫PDFを読み込む
# =========================

def load_cat_pdfs(
    documents_directory: str
) -> list:
    """
    sample_pdfフォルダにある
    猫資料のPDFをすべて読み込む。
    """

    folder_path = Path(
        documents_directory
    )

    # フォルダ確認
    if not folder_path.exists():

        raise FileNotFoundError(
            f"フォルダが見つかりません: "
            f"{folder_path}"
        )

    if not folder_path.is_dir():

        raise NotADirectoryError(
            f"フォルダではありません: "
            f"{folder_path}"
        )

    # PDFだけ取得
    pdf_files = [
        file
        for file in folder_path.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".pdf"
    ]

    # ファイル名順
    pdf_files.sort(
        key=lambda file: file.name
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"PDFが見つかりません: "
            f"{folder_path}"
        )

    all_pages = []

    print(
        "=============================="
    )

    print(
        "猫RAG用PDFを読み込みます"
    )

    print(
        "=============================="
    )

    print(
        f"\nPDFファイル数: "
        f"{len(pdf_files)}"
    )

    # =========================
    # PDFを1つずつ読み込む
    # =========================

    for pdf_file in pdf_files:

        print(
            f"\n読み込み中: "
            f"{pdf_file.name}"
        )

        try:

            loader = PyPDFLoader(
                str(pdf_file)
            )

            pages = loader.load()

        except Exception as error:

            print(
                f"読み込み失敗: "
                f"{pdf_file.name}"
            )

            print(
                f"理由: {error}"
            )

            continue

        valid_page_count = 0

        # =========================
        # 各ページを処理
        # =========================

        for page in pages:

            # ★重要
            # PDFから取得した文章を
            # Chromaへ入れる前にきれいにする

            cleaned_text = clean_pdf_text(
                page.page_content
            )

            # 空ページは除外
            if not cleaned_text:

                continue

            # きれいにした文章へ置き換える
            page.page_content = (
                cleaned_text
            )

            # ファイル名を保存
            page.metadata[
                "source_file"
            ] = pdf_file.name

            # ファイルパスを保存
            page.metadata[
                "source_path"
            ] = str(pdf_file)

            all_pages.append(
                page
            )

            valid_page_count += 1

        print(
            f"読み込み完了: "
            f"{valid_page_count}ページ"
        )

    # =========================
    # 読み込み結果確認
    # =========================

    if not all_pages:

        raise ValueError(
            "読み込める文章情報が"
            "ありませんでした。"
        )

    print(
        f"\n全PDFの合計ページ数: "
        f"{len(all_pages)}"
    )

    return all_pages


# =========================
# 文章をチャンクに分割
# =========================

def split_documents(
    pages: list
) -> list:
    """
    PDFの文章を検索しやすい大きさへ
    分割する。
    """

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "、",
                " ",
                ""
            ]
        )
    )

    chunks = (
        text_splitter.split_documents(
            pages
        )
    )

    print(
        f"\nチャンク数: "
        f"{len(chunks)}"
    )

    return chunks


# =========================
# Chromaデータベース作成
# =========================

def create_database(
    chunks: list
) -> None:
    """
    猫資料をEmbeddingして
    Chromaデータベースへ保存する。
    """

    database_path = Path(
        DATABASE_DIRECTORY
    )

    # =========================
    # 古いDBを削除
    # =========================

    if database_path.exists():

        print(
            "\n古いChromaデータベースを"
            "削除します。"
        )

        shutil.rmtree(
            database_path
        )

    # =========================
    # Embeddingモデル
    # =========================

    print(
        "\nEmbeddingモデルを"
        "読み込んでいます..."
    )

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=(
                EMBEDDING_MODEL_NAME
            )
        )
    )

    # =========================
    # Chromaへ登録
    # =========================

    print(
        "猫資料をChromaデータベースへ"
        "登録しています..."
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=(
            DATABASE_DIRECTORY
        )
    )

    print(
        "\nChromaデータベースを"
        "作成しました。"
    )


# =========================
# メイン処理
# =========================

def main():

    try:

        print(
            "\n=============================="
        )

        print(
            "猫RAG用データベースを"
            "作成します"
        )

        print(
            "=============================="
        )

        # PDF読み込み
        pages = load_cat_pdfs(
            PDF_DIRECTORY
        )

        # チャンク分割
        chunks = split_documents(
            pages
        )

        # Chroma作成
        create_database(
            chunks
        )

        print(
            "\n=============================="
        )

        print(
            "データベース作成完了です。"
        )

        print(
            "=============================="
        )

    except Exception as error:

        print(
            "\nエラーが発生しました。"
        )

        print(
            f"詳細: {error}"
        )


# =========================
# 起動
# =========================

if __name__ == "__main__":
    main()
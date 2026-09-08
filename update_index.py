import hashlib
import json
import os
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from embeddings import Qwen3Embeddings

import config

MARKDOWN_DIR = config.path("paths.data_dir")
FAISS_INDEX_PATH = str(config.path("paths.faiss_index"))
MANIFEST_PATH = config.path("paths.manifest")

# 与 01_langchain_example.py 保持完全一致，否则新老向量不在同一语义空间
embeddings = Qwen3Embeddings(
    model_path=str(config.path("embedding.model_path")),
    device=config.get("embedding.device", "cuda"),
    max_length=config.get("embedding.max_length", 2048),
    batch_size=config.get("embedding.batch_size", 16),
)

CHUNK_SIZE = config.get("chunking.chunk_size", 1200)
CHUNK_OVERLAP = config.get("chunking.chunk_overlap", 300)

HEADERS_TO_SPLIT_ON = config.headers()

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=True,  # 标题存入 metadata，正文保持纯净，避免与前置标题重复
)

sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def header_prefix(metadata: dict) -> str:
    """从 metadata 重建完整章节标题前缀，保证每个块自带章节上下文。"""
    lines = [
        f"{mark} {metadata[key]}"
        for mark, key in HEADERS_TO_SPLIT_ON
        if metadata.get(key)
    ]
    return "\n".join(lines)


def add_prefix(doc, prefix: str):
    if prefix:
        doc.page_content = f"{prefix}\n{doc.page_content}"
    return doc


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    current = {p.name: file_md5(p) for p in md_files}

    manifest = load_manifest()
    index_exists = os.path.exists(FAISS_INDEX_PATH)
    first_run = not MANIFEST_PATH.exists()

    if first_run:
        # 旧索引（由 01 脚本构建）没有 stable id 和清单，无法增量比对，全量重建建立基线
        if index_exists:
            shutil.rmtree(FAISS_INDEX_PATH)
            print("首次运行：清理旧索引，全量重建以建立清单基线。")
        vectorstore = None
        to_add = list(current)
        removed = []
        modified = []
    else:
        vectorstore = (
            FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            if index_exists
            else None
        )
        removed = [n for n in manifest if n not in current]
        modified = [n for n in current if n in manifest and manifest[n]["md5"] != current[n]]
        to_add = [n for n in current if n not in manifest] + modified

    to_delete = removed + modified

    if not to_add and not to_delete:
        print("知识库已是最新，无需更新。")
        return

    # 先删旧向量（stable id = 文件名::块序号）
    if vectorstore is not None:
        for name in to_delete:
            old_count = manifest[name]["chunk_count"]
            vectorstore.delete([f"{name}::{i}" for i in range(old_count)])
            print(f"[删除] {name}（{old_count} 个旧块）")

    # 再追加新向量
    for name in to_add:
        text = (MARKDOWN_DIR / name).read_text(encoding="utf-8")
        md_docs = markdown_splitter.split_text(text)
        chunks = []
        for doc in md_docs:
            doc.metadata["source"] = name
            prefix = header_prefix(doc.metadata)
            if len(doc.page_content) <= CHUNK_SIZE:
                chunks.append(add_prefix(doc, prefix))
            else:
                subs = sub_splitter.split_documents([doc])
                for sub in subs:
                    sub.metadata["source"] = name
                    chunks.append(add_prefix(sub, prefix))
        ids = [f"{name}::{i}" for i in range(len(chunks))]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings, ids=ids)
        else:
            vectorstore.add_documents(chunks, ids=ids)
        manifest[name] = {"md5": current[name], "chunk_count": len(chunks)}
        print(f"[新增] {name}（{len(chunks)} 个块）")

    # 从清单移除已从目录删除的文件
    for name in removed:
        manifest.pop(name, None)

    vectorstore.save_local(FAISS_INDEX_PATH)
    save_manifest(manifest)
    print("更新完成。")


if __name__ == "__main__":
    main()

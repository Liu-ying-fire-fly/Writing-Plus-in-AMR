# view_chunks.py
from embeddings import Qwen3Embeddings
from langchain_community.vectorstores import FAISS

embeddings = Qwen3Embeddings(
    model_path="./model/Qwen3-Embedding-0.6B-modelscope",
    device="cuda",
    max_length=2048,
    batch_size=16,
)

vectorstore = FAISS.load_local(
    "paper_vertical", embeddings, allow_dangerous_deserialization=True
)

# docstore 里 key 是 "文件名::块序号"，value 是 Document
for doc_id, doc in vectorstore.docstore._dict.items():
    source = doc.metadata.get("source", "?")
    print(f"===== {doc_id}  (source={source}, {len(doc.page_content)} 字符) =====")
    print(doc.page_content)
    print()

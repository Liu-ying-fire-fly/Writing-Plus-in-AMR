"""阶段5：合并去重 + 精炼生成。

流程：
1. 复用阶段4 的 GraphRetriever 拿到合并去重后的 chunk（向量 + 图扩展）；
2. cross-encoder 重排：对每个 chunk 计算 query-chunk 相关度，统一重排（向量块的 L2 距离
   与图块缺失相似度的问题，在这里被统一到同一标尺）；
3. 两阶段精炼生成：
   - 摘要压缩（map）：并行把每个 chunk 压缩成与 query 相关的要点，并编号引用；
   - 生成（reduce）：汇总要点，生成带 [1][2] 编号引用的最终答案，文末附来源。
"""

import argparse
import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from neo4j import GraphDatabase
from sentence_transformers import CrossEncoder

import config

load_dotenv()

# 国内环境走 HF 镜像下载 reranker 模型
os.environ.setdefault("HF_ENDPOINT", config.get("hf_endpoint", "https://hf-mirror.com"))

# 复用阶段4 的向量召回 + 图遍历扩展
g4 = importlib.import_module("04_graph_retrieval")

# 多语 cross-encoder，兼容中文 query + 英文论文
RERANKER_MODEL = config.get("reranker.model", "BAAI/bge-reranker-v2-m3")

LLM_MODEL = config.get("deepseek.model")
LLM_BASE_URL = config.get("deepseek.base_url")

# ========== 提示词 ==========

COMPRESSION_PROMPT = """你是论文写作助手。下面是从本地论文库检索到的一段资料（编号 [{idx}]，来源 {source}）。
请提取与写作任务直接相关的关键信息，压缩成要点，要求：
- 每条要点一句话，保留关键方法名、数值、结论；
- 严格基于给定片段，不得编造或补充；
- 若片段与任务无关，只输出「无相关信息」。

写作任务：{query}

资料 [{idx}] 来源：{source}
{text}
"""

GENERATION_PROMPT = """你是自动调制识别（AMR）领域的资深研究员和论文写作助手。
下面是从本地论文库检索并压缩后的资料要点，每条带编号引用来源。
请根据这些要点完成写作任务。

要求：
1. 内容严格基于给定要点，不得编造数据、方法或结论；
2. 正文中用 [1][2] 等编号标注引用来源；
3. 语言学术化、逻辑清晰，适合直接用于论文初稿；
4. 若资料不足以回答，请明确说明。

写作任务：{query}

资料要点：
{points}
"""


# ========== 重排 ==========

def rerank_chunks(query: str, chunks: list, reranker: CrossEncoder):
    """用 cross-encoder 对 query 与每个 chunk 打分，按相关度降序重排。"""
    if not chunks:
        return chunks
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks


# ========== 两阶段生成 ==========

def build_llm(temperature: float, max_tokens: int) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=LLM_BASE_URL,
    )


def compress_chunk(query: str, idx: int, source: str, text: str, llm: ChatDeepSeek) -> str:
    prompt = COMPRESSION_PROMPT.format(query=query, idx=idx, source=source, text=text)
    return llm.invoke(prompt).content.strip()


def compress_all(query: str, top_chunks: list, llm: ChatDeepSeek, max_workers: int = 8) -> list:
    """摘要压缩（map）：并行把每个 chunk 压缩成要点，保序返回。"""
    compressed = [None] * len(top_chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, c in enumerate(top_chunks):
            idx = i + 1
            futures[pool.submit(compress_chunk, query, idx, c["filename"], c["text"], llm)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                compressed[i] = fut.result()
            except Exception as e:
                compressed[i] = f"（压缩失败：{e}）"
    return compressed


def filter_relevant(top_chunks: list, compressed: list):
    """过滤掉「无相关信息」或压缩失败的 chunk，返回 (有效 chunk, 有效要点)。"""
    rel_chunks, rel_points = [], []
    for c, txt in zip(top_chunks, compressed):
        if txt.strip().rstrip("。.") == "无相关信息" or txt.startswith("（压缩失败"):
            continue
        rel_chunks.append(c)
        rel_points.append(txt)
    return rel_chunks, rel_points


def generate_answer(query: str, top_chunks: list, compressed: list, llm: ChatDeepSeek) -> str:
    """生成（reduce）：汇总压缩要点，产出带引用的最终答案。"""
    parts = [
        f"[{i + 1}] 来源：{c['filename']}\n{txt}"
        for i, (c, txt) in enumerate(zip(top_chunks, compressed))
    ]
    points = "\n\n".join(parts)
    prompt = GENERATION_PROMPT.format(query=query, points=points)
    return llm.invoke(prompt).content


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser(description="阶段5：合并去重 + 精炼生成")
    parser.add_argument("--query", default="小样本识别的最好性能是哪篇论文，最高识别准确率达到了多少？")
    parser.add_argument("--k", type=int, default=config.get("retrieval.k", 5), help="向量召回 top-K")
    parser.add_argument("--hops", type=int, default=config.get("retrieval.hops", 2), help="图遍历跳数")
    parser.add_argument("--top-n", type=int, default=config.get("refine.top_n", 10), help="重排后送入压缩的 chunk 数")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--max-workers", type=int, default=config.get("refine.max_workers", 4), help="压缩并发数")
    parser.add_argument("--show-points", action="store_true", help="打印摘要压缩的中间要点")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", config.get("neo4j.uri", "bolt://localhost:7687")))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", config.get("neo4j.user", "neo4j")))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", config.get("neo4j.database", "neo4j")))
    args = parser.parse_args()

    if not args.password:
        print("未提供 NEO4J_PASSWORD，请通过环境变量或 --password 传入。")
        return

    print("加载 cross-encoder 重排器 ...")
    reranker = CrossEncoder(args.reranker_model, max_length=config.get("reranker.max_length", 512))

    print("加载向量库 ...")
    vectorstore = g4.build_vectorstore()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    retriever = g4.GraphRetriever(vectorstore, driver, database=args.database)

    print(f"检索 + 图扩展 ...")
    result = retriever.retrieve(query=args.query, k=args.k, hops=args.hops)
    chunks = result["chunks"]

    if not chunks:
        print("未检索到相关内容。")
        driver.close()
        return

    print(f"重排 {len(chunks)} 块，取 top-{args.top_n} 精炼 ...")
    chunks = rerank_chunks(args.query, chunks, reranker)
    top_chunks = chunks[: args.top_n]

    compression_llm = build_llm(
        temperature=config.get("refine.compression_temperature", 0.0),
        max_tokens=config.get("refine.compression_max_tokens", 2048),
    )
    generation_llm = build_llm(
        temperature=config.get("refine.generation_temperature", 0.7),
        max_tokens=config.get("refine.generation_max_tokens", 8192),
    )

    compressed = compress_all(args.query, top_chunks, compression_llm, max_workers=args.max_workers)

    if args.show_points:
        print("\n" + "-" * 60)
        print("摘要压缩要点（top-{} 全部）：".format(args.top_n))
        for i, (c, txt) in enumerate(zip(top_chunks, compressed), 1):
            print(f"  [{i}] {c['filename']}:\n      {txt}\n")

    rel_chunks, rel_points = filter_relevant(top_chunks, compressed)
    if not rel_chunks:
        print("\n检索到的资料不足以回答该问题（压缩后无相关信息）。")
        driver.close()
        return

    answer = generate_answer(args.query, rel_chunks, rel_points, generation_llm)

    print("\n" + "=" * 60)
    print("写作任务：", args.query)
    print("=" * 60)
    print(answer)
    print("\n" + "-" * 60)
    print("引用来源：")
    for i, c in enumerate(rel_chunks, 1):
        score = c.get("rerank_score")
        print(f"  [{i}] {c['filename']}  (chunk {c['chunk_id']}, 相关度 {score:.4f})")

    driver.close()


if __name__ == "__main__":
    main()

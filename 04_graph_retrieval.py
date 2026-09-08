"""阶段4：检索时的图遍历扩展。

query 进来后：
1. 向量检索召回 top-K 个 chunk（FAISS，通过 index_to_docstore_id 取回 chunk_id）；
2. 从这些 chunk 提取命中的实体节点（Neo4j 溯源边 Entity-[:MENTIONED_IN]->Chunk）；
3. 从这些实体出发，沿语义边无向走 1~2 跳，找到邻域实体
   （命中 "LoRA" 可扩展出同论文/同数据集的 "PEFT"、任务、数据集等）；
4. 把扩展实体映射的原文 chunk 捞出来，与向量结果按 chunk_id 去重合并。

语义边 = USES/TARGETS/ADDRESSES/EVALUATED_ON/REPORTED/APPLIES_TO/INVOLVES，
排除溯源边 PART_OF 与 MENTIONED_IN，避免遍历漏进 Chunk 节点导致无限扩张。

返回结构（供阶段5精炼层使用）：
{"chunks": [...], "seeds": [...], "expanded": [...], ...}
每个 chunk：{"chunk_id", "text", "filename", "source": "vector"|"graph",
             "score"? , "hop"?, "n_seeds"?, "entity"?}
"""

import argparse
import os

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.vectorstores import FAISS

from embeddings import Qwen3Embeddings

import config

load_dotenv()

FAISS_INDEX_PATH = str(config.path("paths.faiss_index"))
MODEL_PATH = str(config.path("embedding.model_path"))

# 语义边（用于图遍历扩展）；排除溯源边 PART_OF / MENTIONED_IN
SEMANTIC_REL_TYPES = config.get("retrieval.semantic_rel_types", [])


def build_vectorstore() -> FAISS:
    embeddings = Qwen3Embeddings(
        model_path=MODEL_PATH,
        device=config.get("embedding.device", "cuda"),
        max_length=config.get("embedding.max_length", 2048),
        batch_size=config.get("embedding.batch_size", 16),
    )
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def vector_search(vectorstore: FAISS, query: str, k: int):
    """向量召回 top-K，返回 [{chunk_id, score, text}]，score 为 L2 距离（越小越相似）。"""
    embedding = vectorstore.embedding_function.embed_query(query)
    vector = np.array([embedding], dtype=np.float32)
    if getattr(vectorstore, "_normalize_L2", False):
        vector = vector / np.linalg.norm(vector)
    scores, indices = vectorstore.index.search(vector, k)
    out = []
    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)
        if idx == -1:
            continue
        chunk_id = vectorstore.index_to_docstore_id[idx]
        doc = vectorstore.docstore.search(chunk_id)
        out.append({"chunk_id": chunk_id, "score": float(score), "text": doc.page_content})
    return out


class GraphRetriever:
    """向量召回 + 图遍历扩展的检索器。"""

    def __init__(self, vectorstore: FAISS, driver, database: str = "neo4j"):
        self.vectorstore = vectorstore
        self.driver = driver
        self.database = database

    def _run(self, cypher: str, **params):
        with self.driver.session(database=self.database) as session:
            return list(session.run(cypher, **params))

    def seed_entities(self, chunk_ids):
        """从向量召回的 chunk 提取命中实体（溯源边 MENTIONED_IN）。"""
        if not chunk_ids:
            return []
        rows = self._run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {chunk_id: cid})
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            chunk_ids=chunk_ids,
        )
        return [(r["name"], r["type"]) for r in rows]

    def expand_entities(self, seeds, hops: int = 2):
        """从种子实体沿语义边无向走 1~hops 跳，返回邻域实体（已排除种子）。

        每行含 hop（最短跳数）与 n_seeds（该实体与多少个种子实体相连，作共现强度）。
        """
        if not seeds:
            return []
        seed_params = [{"name": n, "type": t} for n, t in seeds]
        hops = max(1, min(int(hops), 2))  # 变长路径上界需写成字面量，限制在 1~2
        rows = self._run(
            f"""
            UNWIND $seeds AS s
            MATCH (seed:Entity {{name: s.name, type: s.type}})
            MATCH p = (seed)-[rels*1..{hops}]-(e:Entity)
            WHERE ALL(r IN rels WHERE type(r) IN $rel_types)
            WITH e, min(length(p)) AS hop, count(DISTINCT seed) AS n_seeds
            RETURN e.name AS name, e.type AS type, hop, n_seeds
            ORDER BY hop ASC, n_seeds DESC, name ASC
            """,
            seeds=seed_params,
            rel_types=SEMANTIC_REL_TYPES,
        )
        seed_set = set(seeds)
        return [
            {"name": r["name"], "type": r["type"], "hop": r["hop"], "n_seeds": r["n_seeds"]}
            for r in rows
            if (r["name"], r["type"]) not in seed_set
        ]

    def fetch_chunks(self, entities):
        """取扩展实体映射的原文 chunk（溯源边 MENTIONED_IN）。"""
        if not entities:
            return []
        entity_params = [{"name": e["name"], "type": e["type"]} for e in entities]
        rows = self._run(
            """
            UNWIND $entities AS x
            MATCH (e:Entity {name: x.name, type: x.type})-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.chunk_id AS chunk_id, c.text AS text, e.name AS name, e.type AS type
            """,
            entities=entity_params,
        )
        return [
            {"chunk_id": r["chunk_id"], "text": r["text"], "name": r["name"], "type": r["type"]}
            for r in rows
        ]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        hops: int = 2,
        max_expanded_entities: int = 40,
        max_expanded_chunks: int = 20,
        max_chunks_per_entity: int = 3,
    ):
        """完整检索：向量召回 + 图扩展，返回去重合并后的 chunk 列表。"""
        # 1. 向量召回
        vec_chunks = vector_search(self.vectorstore, query, k)
        for c in vec_chunks:
            c["filename"] = c["chunk_id"].rsplit("::", 1)[0]
            c["source"] = "vector"

        # 2. 种子实体
        seeds = self.seed_entities([c["chunk_id"] for c in vec_chunks])

        # 3. 图扩展（跳数+共现强度排序，去种子，限实体数）
        expanded = self.expand_entities(seeds, hops=hops)[:max_expanded_entities]

        # 4. 扩展实体映射的 chunk：按实体排序优先级 + 去重 + 每实体上限 + 限总量
        raw_graph_chunks = self.fetch_chunks(expanded)
        rank = {(e["name"], e["type"]): i for i, e in enumerate(expanded)}
        graph_chunks = []
        seen = set()
        per_entity = {}
        for gc in sorted(
            raw_graph_chunks,
            key=lambda c: (rank[(c["name"], c["type"])], c["chunk_id"]),
        ):
            if gc["chunk_id"] in seen:
                continue
            key = (gc["name"], gc["type"])
            if per_entity.get(key, 0) >= max_chunks_per_entity:
                continue
            seen.add(gc["chunk_id"])
            per_entity[key] = per_entity.get(key, 0) + 1
            e = expanded[rank[key]]
            graph_chunks.append({
                "chunk_id": gc["chunk_id"],
                "text": gc["text"],
                "filename": gc["chunk_id"].rsplit("::", 1)[0],
                "source": "graph",
                "hop": e["hop"],
                "n_seeds": e["n_seeds"],
                "entity": gc["name"],
            })
            if len(graph_chunks) >= max_expanded_chunks:
                break

        # 5. 合并去重：向量结果优先，图扩展结果去重后追加
        merged = []
        seen_ids = set()
        for c in vec_chunks + graph_chunks:
            if c["chunk_id"] in seen_ids:
                continue
            seen_ids.add(c["chunk_id"])
            merged.append(c)

        return {
            "chunks": merged,
            "seeds": seeds,
            "expanded": expanded,
            "vector_count": len(vec_chunks),
            "graph_count": len(graph_chunks),
        }


def main():
    parser = argparse.ArgumentParser(description="阶段4：检索时的图遍历扩展")
    parser.add_argument("--query", default="What is automatic modulation recognition?")
    parser.add_argument("--k", type=int, default=config.get("retrieval.k", 3), help="向量召回 top-K")
    parser.add_argument("--hops", type=int, default=config.get("retrieval.hops", 2), help="图遍历跳数（1 或 2）")
    parser.add_argument("--max-expanded-entities", type=int, default=config.get("retrieval.max_expanded_entities", 40), help="扩展实体数量上限")
    parser.add_argument("--max-expanded-chunks", type=int, default=config.get("retrieval.max_expanded_chunks", 20), help="扩展 chunk 数量上限")
    parser.add_argument("--max-chunks-per-entity", type=int, default=config.get("retrieval.max_chunks_per_entity", 3), help="单个扩展实体贡献的 chunk 上限")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", config.get("neo4j.uri", "bolt://localhost:7687")))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", config.get("neo4j.user", "neo4j")))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", config.get("neo4j.database", "neo4j")))
    args = parser.parse_args()

    if not args.password:
        print("未提供 NEO4J_PASSWORD，请通过环境变量或 --password 传入。")
        return

    print("加载向量库 ...")
    vectorstore = build_vectorstore()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()

    retriever = GraphRetriever(vectorstore, driver, database=args.database)
    result = retriever.retrieve(
        args.query,
        k=args.k,
        hops=args.hops,
        max_expanded_entities=args.max_expanded_entities,
        max_expanded_chunks=args.max_expanded_chunks,
        max_chunks_per_entity=args.max_chunks_per_entity,
    )

    print(f"\nQuery: {args.query}")
    print(
        f"向量召回 {result['vector_count']} 块；命中种子实体 {len(result['seeds'])} 个；"
        f"扩展邻域实体 {len(result['expanded'])} 个；图扩展引入 chunk {result['graph_count']} 个；"
        f"合并去重后共 {len(result['chunks'])} 块。\n"
    )

    for c in result["chunks"]:
        if c["source"] == "vector":
            tag = f"向量 score={c['score']:.4f}"
        else:
            tag = f"图 hop={c['hop']} n_seeds={c['n_seeds']} entity={c['entity']}"
        snippet = c["text"].replace("\n", " ")[:100]
        print(f"  [{tag}] {c['chunk_id']}\n      {snippet}...")

    driver.close()


if __name__ == "__main__":
    main()

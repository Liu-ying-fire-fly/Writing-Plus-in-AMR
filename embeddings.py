# amr_rag/embeddings.py
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings


class Qwen3Embeddings(Embeddings):
    """
    Qwen3-Embedding-0.6B 本地嵌入类
    支持 query / document 不同的指令前缀，使用 last token pooling
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 32,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        # Qwen3 官方非对称检索指令前缀
        self.query_instruction = (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        )

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Qwen3 系列使用 last token pooling"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """批量编码文本，is_query=True 时添加查询指令前缀"""
        if is_query:
            texts = [self.query_instruction + t for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档块编码：不加指令"""
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        """查询编码：加检索指令"""
        return self._encode([text], is_query=True)[0]

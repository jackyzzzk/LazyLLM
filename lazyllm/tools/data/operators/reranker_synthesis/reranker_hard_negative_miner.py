"""
Reranker Hard Negative Miner Operator

This operator mines hard negative samples specifically optimized for reranker model training.
该算子专门为 Reranker 模型训练挖掘困难负样本。
"""
import random
import numpy as np
from typing import List, Optional
from lazyllm.thirdparty import numpy as np, bm25s, jieba, Stemmer
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 获取或创建 reranker 分组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


class RerankerHardNegativeMiner(reranker):
    """
    Mine hard negative samples optimized for reranker training.
    为 Reranker 训练挖掘优化的困难负样本。

    Reranker models benefit from carefully selected hard negatives that are:
    - Semantically similar to positive but not correct
    - Lexically similar but semantically different
    - From the same domain but different topics

    Mining strategies:
    - random: Random sampling from corpus (baseline)
    - bm25: BM25-based lexical similarity (good for lexical confusers)
    - semantic: Embedding-based semantic similarity (good for semantic confusers)
    - mixed: Combination of BM25 and semantic negatives

    This operator uses forward_batch_input because it needs to:
    1. Build a global corpus from all positive samples
    2. Build BM25 index once for the entire corpus
    3. Compute corpus embeddings once for semantic mining

    Args:
        mining_strategy: Strategy for mining negatives ("random", "bm25", "semantic", "mixed")
        num_negatives: Number of negative samples per query (default: 7, for train_group_size=8)
        embedding_serving: Embedding service for semantic mining (optional)
        bm25_ratio: Ratio of BM25 negatives in mixed mode (default: 0.5)
        seed: Random seed for reproducibility
    """

    def __init__(
            self,
            mining_strategy: str = "random",
            num_negatives: int = 7,
            embedding_serving=None,
            bm25_ratio: float = 0.5,
            seed: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mining_strategy = mining_strategy
        self.num_negatives = num_negatives
        self.embedding_serving = embedding_serving
        self.bm25_ratio = bm25_ratio
        self.seed = seed
        self._bm25 = None
        self._corpus = None
        self._corpus_embeddings = None
        LOG.info(f"Initializing {self.__class__.__name__} with strategy: {mining_strategy}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerHardNegativeMiner 算子用于挖掘 Reranker 训练的困难负样本。\n\n"
                "核心功能：\n"
                "- 支持多种负样本挖掘策略（随机、BM25、语义、混合）\n"
                "- 混合模式结合词汇和语义相似度\n"
                "- 针对 Reranker 训练优化的负样本选择\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- input_pos_key: 正样本字段名（默认：'pos'）\n"
                "- output_neg_key: 输出负样本字段名（默认：'neg'）\n\n"
                "输出：包含困难负样本的数据列表"
            )
        else:
            return (
                "RerankerHardNegativeMiner mines hard negatives optimized for reranker training.\n\n"
                "Features:\n"
                "- Multiple mining strategies (random, BM25, semantic, mixed)\n"
                "- Mixed mode combines lexical and semantic similarity\n"
                "- Negative selection optimized for reranker training"
            )

    def _init_bm25(self, corpus: List[str]):
        """Initialize BM25 index for lexical similarity mining.
        """
        LOG.info(f"Initializing BM25 index for {len(corpus)} documents...")
        self._corpus = corpus
        corpus_tokens = bm25s.tokenize(
            [self._tokenizer(doc) for doc in corpus],
            stopwords=self._stopwords,
            stemmer=self._stemmer,
        )
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)
        LOG.info("BM25 index initialized successfully.")


    def _compute_corpus_embeddings(self, corpus: List[str]):
        """Compute embeddings for the entire corpus."""
        if self.embedding_serving is None:
            raise ValueError("Embedding serving is required for semantic mining strategy")

        LOG.info(f"Computing embeddings for {len(corpus)} documents...")
        self._corpus_embeddings = self.embedding_serving(corpus)
        self._corpus_embeddings = np.array(self._corpus_embeddings)
        self._corpus = corpus
        LOG.info("Corpus embeddings computed successfully.")

    def _mine_random(self, query: str, pos_set: set, corpus: List[str], num: int) -> List[str]:
        """Mine negatives using random sampling."""
        candidates = [doc for doc in corpus if doc not in pos_set]
        if len(candidates) <= num:
            return candidates
        return random.sample(candidates, num)

    def _mine_bm25(self, query: str, pos_set: set, corpus: List[str]) -> List[str]:
        """Mine negatives using BM25 similarity.
        
        """
        if self._bm25 is None or self._corpus != corpus:
            self._init_bm25(corpus)

        # Tokenize query using the same method as corpus
        tokenized_query = bm25s.tokenize(
            self._tokenizer(query),
            stopwords=self._stopwords,
            stemmer=self._stemmer
        )

        # Retrieve all documents with scores
        k = min(len(corpus), self.num_negatives + len(pos_set) + 10)  # Get more than needed
        indices, scores = self._bm25.retrieve(tokenized_query, k=k)

        # Filter out positives and take top-k
        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc = corpus[idx]
            if doc not in pos_set:
                results.append(doc)
                if len(results) >= self.num_negatives:
                    break

        return results

    def _cosine_similarity(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and corpus.

        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Normalize corpus
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_norms = np.where(corpus_norms > 0, corpus_norms, 1)  # Avoid division by zero
        corpus_normalized = corpus_embeddings / corpus_norms

        # Compute cosine similarity
        similarities = np.dot(corpus_normalized, query_embedding)
        return similarities

    def _mine_semantic(self, query: str, pos_set: set, corpus: List[str]) -> List[str]:
        """Mine negatives using semantic similarity.
        """
        if self._corpus_embeddings is None or self._corpus != corpus:
            self._compute_corpus_embeddings(corpus)

        # Get query embedding
        query_embedding = self.embedding_serving([query])[0]
        query_embedding = np.array(query_embedding)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self._corpus_embeddings)

        # Sort by similarity descending, filter out positives
        scored_docs = [(sim, doc) for sim, doc in zip(similarities, corpus) if doc not in pos_set]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top-k as hard negatives (high similarity but not positive)
        return [doc for _, doc in scored_docs[:self.num_negatives]]


    def _mine_mixed(self, query: str, pos_set: set, corpus: List[str], num: int) -> List[str]:
        """Mine negatives using mixed strategy (BM25 + semantic)."""
        num_bm25 = max(1, int(num * self.bm25_ratio))
        num_semantic = num - num_bm25

        bm25_negs = self._mine_bm25(query, pos_set, corpus, num_bm25)
        # Update pos_set to exclude BM25 negatives for semantic mining
        pos_set_extended = pos_set | set(bm25_negs)
        semantic_negs = self._mine_semantic(query, pos_set_extended, corpus, num_semantic)

        return bm25_negs + semantic_negs

    def forward_batch_input(
            self,
            data: List[dict],
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            output_neg_key: str = "neg",
            corpus: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Mine hard negative samples for each query.

        Args:
            data: List of dict with query-pos pairs
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field (list of passages)
            output_neg_key: Key for output negative samples field
            corpus: Optional external corpus for mining negatives

        Returns:
            List of dict with mined hard negatives added
        """
        assert isinstance(data, list), "Input data must be a list"

        # Build corpus from data if not provided
        if corpus is None:
            # Extract unique passages from all positive samples
            all_passages = []
            for item in data:
                pos_list = item.get(input_pos_key, [])
                if isinstance(pos_list, list):
                    all_passages.extend(pos_list)
                else:
                    all_passages.append(pos_list)
            corpus = list(set(all_passages))

        LOG.info(f"Mining hard negatives for {len(data)} queries from corpus of {len(corpus)} documents...")

        # Set random seed for reproducibility
        random.seed(self.seed)

        # Select mining function
        if self.mining_strategy == "random":
            mine_func = lambda q, ps, c: self._mine_random(q, ps, c, self.num_negatives)
        elif self.mining_strategy == "bm25":
            mine_func = lambda q, ps, c: self._mine_bm25(q, ps, c, self.num_negatives)
        elif self.mining_strategy == "semantic":
            mine_func = lambda q, ps, c: self._mine_semantic(q, ps, c, self.num_negatives)
        elif self.mining_strategy == "mixed":
            mine_func = lambda q, ps, c: self._mine_mixed(q, ps, c, self.num_negatives)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

        # Mine negatives for each query
        results = []
        for item in data:
            query = item.get(input_query_key, "")
            pos_samples = item.get(input_pos_key, [])

            # Convert to set for fast lookup
            if isinstance(pos_samples, list):
                pos_set = set(pos_samples)
            else:
                pos_set = {pos_samples}

            # Mine hard negatives
            negatives = mine_func(query, pos_set, corpus)

            new_row = item.copy()
            new_row[output_neg_key] = negatives
            results.append(new_row)

        LOG.info(f"Hard negative mining completed for {len(results)} samples.")
        return results

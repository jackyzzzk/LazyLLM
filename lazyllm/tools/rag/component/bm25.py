from typing import List, Tuple, Optional
from ..doc_node import DocNode
from lazyllm.thirdparty import jieba, bm25s, Stemmer
from .stopwords import STOPWORDS_CHINESE


class BM25:
    """基于 BM25 算法实现的检索器，用于从节点集合中根据查询词检索最相关的文本节点。

Args:
    nodes (List[DocNode]): 需要建立索引的文本节点列表。
    language (str): 所使用的语言，支持 ``en``（英文）或 ``zh``（中文）。默认为 ``en``。
    topk (int): 每次检索返回的最大节点数量，默认值为2。
    **kwargs: 其他参数。
"""

    def __init__(
        self,
        nodes: List[DocNode],
        language: str = 'en',
        topk: int = 2,
        **kwargs,
    ) -> None:
        if language == 'en':
            self._stemmer = Stemmer.Stemmer('english')
            self._stopwords = language
            self._tokenizer = lambda t: t
        elif language == 'zh':
            self._stemmer = None
            # TODO(ywt): after bm25s supports cn stopwards, update this
            self._stopwords = STOPWORDS_CHINESE
            self._tokenizer = lambda t: ' '.join(jieba.lcut(t))
        self.topk = min(topk, len(nodes))
        self.nodes = nodes

        corpus_tokens = bm25s.tokenize(
            [self._tokenizer(node.get_text()) for node in nodes],
            stopwords=self._stopwords,
            stemmer=self._stemmer,
        )
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

    def retrieve(self, query: str, topk: Optional[int] = None) -> List[Tuple[DocNode, float]]:
        """使用BM25算法检索与查询最相关的文档节点。

Args:
    query (str): 查询文本。

**Returns:**

- List[Tuple[DocNode, float]]: 返回一个列表，每个元素为(文档节点, 相关度分数)的元组。
"""
        if topk is None:
            topk = self.topk
        else:
            topk = min(topk, len(self.nodes))
        tokenized_query = bm25s.tokenize(
            self._tokenizer(query), stopwords=self._stopwords, stemmer=self._stemmer
        )
        indexs, scores = self.bm25.retrieve(tokenized_query, k=topk)
        results = []
        for idx, score in zip(indexs[0], scores[0]):
            results.append((self.nodes[idx], score))
        return results

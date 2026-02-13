import importlib.util

from functools import lru_cache
from typing import Callable, List, Optional, Union

import lazyllm
from lazyllm.thirdparty import spacy
from lazyllm import ModuleBase, LOG
from .doc_node import DocNode, MetadataMode
from .retriever import _PostProcess


class Reranker(ModuleBase, _PostProcess):
    """用于创建节点（文档）后处理和重排序的模块。

Args:
    name: 用于后处理和重排序过程的排序器类型。默认为 'ModuleReranker'。
    target(str):已废弃参数，仅用于提示用户。
    output_format: 代表输出格式，默认为None，可选值有 'content' 和 'dict'，其中 content 对应输出格式为字符串，dict 对应字典。
    join: 是否联合输出的 k 个节点，当输出格式为 content 时，如果设置该值为 True，则输出一个长字符串，如果设置为 False 则输出一个字符串列表，其中每个字符串对应每个节点的文本内容。当输出格式是 dict 时，不能联合输出，此时join默认为False,，将输出一个字典，包括'content、'embedding'、'metadata'三个key。
    kwargs: 传递给重新排序器实例化的其他关键字参数。

详细解释排序器类型

  - Reranker: 实例化一个具有待排序的文档节点node列表和 query的 SentenceTransformerRerank 重排序器。
  - KeywordFilter: 实例化一个具有指定必需和排除关键字的 KeywordNodePostprocessor。它根据这些关键字的存在或缺失来过滤节点。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, Reranker, Retriever, DocNode
    >>> m = lazyllm.OnlineEmbeddingModule()
    >>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
    >>> retriever = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
    >>> reranker = Reranker(DocNode(text=user_data),query="user query")
    >>> ppl = lazyllm.ActionModule(retriever, reranker)
    >>> ppl.start()
    >>> print(ppl("user query"))
    """
    registered_reranker = dict()

    def __new__(cls, name: str = 'ModuleReranker', *args, **kwargs):
        assert name in cls.registered_reranker, f'Reranker: {name} is not registered, please register first.'
        item = cls.registered_reranker[name]
        if isinstance(item, type) and issubclass(item, Reranker):
            return super(Reranker, cls).__new__(item)
        else:
            return super(Reranker, cls).__new__(cls)

    def __init__(self, name: str = 'ModuleReranker', target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__()
        self._name = name
        self._kwargs = kwargs
        lazyllm.deprecated(bool(target), '`target` parameter of reranker')
        _PostProcess.__init__(self, output_format, join)

    def forward(self, nodes: List[DocNode], query: str = '') -> List[DocNode]:
        results = self.registered_reranker[self._name](nodes, query=query, **self._kwargs)
        LOG.debug(f'Rerank use `{self._name}` and get nodes: {results}')
        return self._post_process(results)

    @classmethod
    def register_reranker(
        cls: 'Reranker', func: Optional[Callable] = None, batch: bool = False
    ):
        """是一个类装饰器工厂方法，它的核心作用是为 Reranker 类提供灵活的排序算法注册机制

Args:
    func (Optional[Callable]):  要注册的排序函数或排序器类。当使用装饰器语法(@)时可省略。
    batch (bool):是否批量处理节点。默认为False，表示逐节点处理。


Examples:

    @Reranker.register_reranker
    def my_reranker(node: DocNode, **kwargs):
        return node.score * 0.8  # 自定义分数计算
    """
        def decorator(f):
            if isinstance(f, type):
                cls.registered_reranker[f.__name__] = f
                return f
            else:
                def wrapper(nodes, **kwargs):
                    if batch:
                        return f(nodes, **kwargs)
                    else:
                        results = [f(node, **kwargs) for node in nodes]
                        return [result for result in results if result]

                cls.registered_reranker[f.__name__] = wrapper
                return wrapper

        return decorator(func) if func else decorator


@lru_cache(maxsize=None)
def get_nlp_and_matchers(language):
    nlp = spacy.blank(language)

    spec = importlib.util.find_spec('spacy.matcher')
    if spec is None:
        raise ImportError(
            'Please install spacy to use spacy module. '
            'You can install it with `pip install spacy==3.7.5`'
        )
    matcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher_module)

    required_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    exclude_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    return nlp, required_matcher, exclude_matcher


@Reranker.register_reranker
def KeywordFilter(node: DocNode, required_keys: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None,
                  language: str = 'en', **kwargs) -> Optional[DocNode]:
    assert required_keys or exclude_keys, 'One of required_keys or exclude_keys should be provided'
    nlp, required_matcher, exclude_matcher = get_nlp_and_matchers(language)
    if required_keys:
        required_matcher.add('RequiredKeywords', list(nlp.pipe(required_keys)))
    if exclude_keys:
        exclude_matcher.add('ExcludeKeywords', list(nlp.pipe(exclude_keys)))

    doc = nlp(node.get_text())
    if required_keys and not required_matcher(doc):
        return None
    if exclude_keys and exclude_matcher(doc):
        return None
    return node

@Reranker.register_reranker()
class ModuleReranker(Reranker):
    """使用可训练模块根据查询相关性重新排序文档的重排序器。

ModuleReranker是一个专门的重排序器，利用可训练模型（如BGE-reranker、Cohere rerank等）来提高检索文档的相关性。它接收文档列表和查询，然后返回按相关性分数重新排序的文档。

Args:
    name (str): 重排序器的名称。默认为"ModuleReranker"。
    model (Union[Callable, str]): 重排序模型。可以是模型名称（字符串）或可调用函数。
    target (Optional[str]): 默认为None。
    output_format (Optional[str]): 输出处理格式。默认为None。
    join (Union[bool, str]): 是否连接结果。默认为False。
    **kwargs: 传递给重排序模型模型的附加关键字参数。


Examples:
    >>> from lazyllm.tools.rag.rerank import ModuleReranker, DocNode
    >>> def simple_reranker(query, documents, top_n):
    ...     query_lower = query.lower()
    ...     scores = []
    ...     for i, doc in enumerate(documents):
    ...         score = sum(1 for word in query_lower.split() if word in doc)
    ...         scores.append((i, score))
    ...     scores.sort(key=lambda x: x[1], reverse=True)
    ...     return scores[:top_n]
    >>> reranker = ModuleReranker(
    ...     model=simple_reranker,
    ...     topk=2
    ... )
    >>> docs = [
    ...     DocNode(text="机器学习算法在数据分析中应用广泛"),
    ...     DocNode(text="深度学习模型需要大量训练数据"),
    ...     DocNode(text="自然语言处理技术发展迅速"),
    ...     DocNode(text="计算机视觉在自动驾驶中的应用")
    ... ]
    >>> query = "机器学习"
    >>> results = reranker.forward(docs, query)
    >>> for i, doc in enumerate(results):
    ...     print(f"  {i+1}. : {doc.text}")
    ...     print(f"     相关性分数: {doc.relevance_score:.4f}")
    """

    def __init__(self, name: str = 'ModuleReranker', model: Union[Callable, str] = None, target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__(name, target, output_format, join, **kwargs)
        assert model is not None, 'Reranker model must be specified as a model name or a callable.'
        if isinstance(model, str):
            self._reranker = lazyllm.TrainableModule(model)
        else:
            self._reranker = model

    def forward(self, nodes: List[DocNode], query: str = '') -> List[DocNode]:
        """重排序器的前向传播，根据与查询的相关性重新排序文档。

此方法接收文档列表和查询，然后使用底层重排序模型对文档进行评分和重新排序。文档以MetadataMode.EMBED格式处理，以确保与重排序模型的兼容性。

Args:
    nodes (List[DocNode]): 要重排序的文档节点列表。
    query (str): 用于排序文档的查询字符串。默认为""。

**Returns:**

- List[DocNode]: 按相关性分数重新排序的文档节点列表，添加了relevance_score属性。
"""
        if not nodes:
            return self._post_process([])

        docs = [node.get_text(metadata_mode=MetadataMode.EMBED) for node in nodes]
        top_n = self._kwargs['topk'] if 'topk' in self._kwargs else len(docs)
        sorted_indices = self._reranker(query, documents=docs, top_n=top_n)
        results = []
        for index, relevance_score in sorted_indices:
            results.append(nodes[index].with_score(relevance_score))
        LOG.debug(f'Rerank use `{self._name}` and get nodes: {results}')
        return self._post_process(results)

# User-defined similarity decorator
def register_reranker(func=None, batch=False):
    return Reranker.register_reranker(func, batch)

import os
import fnmatch
import inspect

from typing import Any, Dict, List, Union, Optional, Callable

from ..doc_node import DocNode, QADocNode
from lazyllm import LOG
from .base import NodeTransform

from lazyllm.components import AlpacaPrompter
from dataclasses import dataclass, field

from lazyllm import TrainableModule
from lazyllm.components.formatter import encode_query_with_filepaths
from ..prompts import LLMTransformParserPrompts

@dataclass
class TransformArgs():
    """
文档转换参数容器，用于统一管理文档处理中的各类配置参数。

Args:
    f(Union[str, Callable]):转换函数或注册的函数名。
    trans_node(bool):是否转换节点类型。
    num_workers (int)：控制是否启用多线程（>0 时启用）。
    kwargs(Dict): 传递给转换函数的额外参数。
    pattern(Union[str, Callable[[str], bool]]):文件名/内容匹配模式。


Examples:

    >>> from lazyllm.tools import TransformArgs
    >>> args = TransformArgs(f=lambda text: text.lower(),num_workers=4,pattern=r'.*\.md$')
    >>>config = {'f': 'parse_pdf','kwargs': {'engine': 'pdfminer'},'trans_node': True}
    >>>args = TransformArgs.from_dict(config)
    print(args['f'])
    print(args.get('unknown'))
    """
    f: Union[str, Callable]
    trans_node: Optional[bool] = None
    num_workers: int = 0
    kwargs: Dict = field(default_factory=dict)
    pattern: Optional[Union[str, Callable[[str], bool]]] = None

    @staticmethod
    def from_dict(d):
        return TransformArgs(f=d['f'], trans_node=d.get('trans_node'), num_workers=d.get(
            'num_workers', 0), kwargs=d.get('kwargs', dict()), pattern=d.get('pattern'))

    def __getitem__(self, key):
        if key in self.__dict__: return getattr(self, key)
        raise KeyError(f'Key {key} is not found in transform args')

    def get(self, key):
        if key in self.__dict__: return getattr(self, key)
        return None

def build_nodes_from_splits(
    text_splits: List[str], doc: DocNode, node_group: str
) -> List[DocNode]:
    nodes: List[DocNode] = []
    for text_chunk in text_splits:
        if not text_chunk:
            continue
        node = DocNode(
            text=text_chunk,
            group=node_group,
            parent=doc,
        )
        nodes.append(node)

    doc.children[node_group] = nodes
    return nodes

def make_transform(t: Union[TransformArgs, Dict[str, Any]], group_name: Optional[str] = None) -> NodeTransform:
    if isinstance(t, dict): t = TransformArgs.from_dict(t)
    transform, trans_node, num_workers = t['f'], t['trans_node'], t['num_workers']
    num_workers = dict(num_workers=num_workers) if num_workers > 0 else dict()
    return (transform(**t['kwargs'], **num_workers).with_name(group_name, copy=False) if isinstance(transform, type)
            else transform.with_name(group_name) if isinstance(transform, NodeTransform)
            else FuncNodeTransform(transform, trans_node=trans_node, **num_workers).with_name(group_name, copy=False))


class AdaptiveTransform(NodeTransform):
    """一个灵活的文档转换系统，根据文档模式应用不同的转换策略。

AdaptiveTransform允许您定义多种转换策略，并根据文档的文件路径或自定义模式匹配自动选择适当的转换方法。当您有不同类型的文档需要不同处理方法时，这特别有用。

Args:
    transforms (Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]]): 转换配置列表或单个转换配置。
    num_workers (int, optional): 并行处理的工作线程数。默认为0。


Examples:
    >>> from lazyllm.tools.rag.transform import AdaptiveTransform, DocNode, SentenceSplitter
    >>> doc1 = DocNode(text="这是第一个文档的内容。它包含多个句子。")
    >>> doc2 = DocNode(text="这是第二个文档的内容。")
    >>> transforms = [
    ...     {
    ...         'f': SentenceSplitter,
    ...         'pattern': '*.txt',
    ...         'kwargs': {'chunk_size': 50, 'chunk_overlap': 10}
    ...     },
    ...     {
    ...         'f': SentenceSplitter,
    ...         'pattern': '*.pdf',
    ...         'kwargs': {'chunk_size': 100, 'chunk_overlap': 20}
    ...     }
    ... ]
    >>> adaptive = AdaptiveTransform(transforms)
    >>> results1 = adaptive.transform(doc1)
    >>> print(f"文档1转换结果: {len(results1)} 个块")
    >>> for i, result in enumerate(results1):
    ...     print(f"  块 {i+1}: {result.text}")
    >>> results2 = adaptive.transform(doc2)
    >>> print(f"文档2转换结果: {len(results2)} 个块")
    >>> for i, result in enumerate(results2):
    ...     print(f"  块 {i+1}: {result.text}")      
    """
    def __init__(self, transforms: Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]],
                 num_workers: int = 0):
        super().__init__(num_workers=num_workers)
        if not isinstance(transforms, (tuple, list)): transforms = [transforms]
        self._transformers = [(t.get('pattern'), make_transform(t)) for t in transforms]

    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        """根据模式匹配使用适当的转换策略转换文档。

此方法按顺序评估每个转换配置，并应用第一个匹配文档路径模式的转换。匹配逻辑支持glob模式和自定义可调用函数。

Args:
    document (DocNode): 要转换的文档节点。
    **kwargs: 传递给转换函数的附加关键字参数。

**Returns:**

- List[Union[str, DocNode]]: 转换结果列表（字符串或DocNode对象）。
"""
        if not isinstance(document, DocNode): LOG.warning(f'Invalud document type {type(document)} got')
        for pt, transform in self._transformers:
            if pt and isinstance(pt, str) and not pt.startswith('*'): pt = os.path.join(os.getcwd(), pt)
            if not pt or (callable(pt) and pt(document.docpath)) or (
                    isinstance(pt, str) and fnmatch.fnmatch(document.docpath, pt)):
                return transform(document, **kwargs)
        LOG.warning(f'No transform found for document {document.docpath} with group name `{self._name}`')
        return []


class FuncNodeTransform(NodeTransform):
    """
用于包装用户自定义函数的转换器类。

此包装器支持两种操作模式：
    1. 当 trans_node 为 False（默认）：转换文本字符串
    2. 当 trans_node 为 True：转换 DocNode 对象

包装器可以处理各种函数签名：
    - str -> List[str]: transform=lambda t: t.split('\\n')
    - str -> str: transform=lambda t: t[:3]
    - DocNode -> List[DocNode]: pipeline(lambda x:x, SentenceSplitter)
    - DocNode -> DocNode: pipeline(LLMParser)

Args:
    func (Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]]): 要包装的用户自定义函数。
    trans_node (bool, optional): 确定函数是操作 DocNode 对象（True）还是文本字符串（False）。默认为 None。
    num_workers (int): 控制并行处理的线程/进程数量。默认为 0。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools.rag import FuncNodeTransform
    >>> from lazyllm.tools import Document, SentenceSplitter

    # Example 1: Text-based transformation (trans_node=False)
    >>> def split_by_comma(text):
    ...     return text.split(',')
    >>> text_transform = FuncNodeTransform(split_by_comma, trans_node=False)

    # Example 2: Node-based transformation (trans_node=True)
    >>> def custom_node_transform(node):
    ...     # Process the DocNode and return a list of DocNodes
    ...     return [node]  # Simple pass-through
    >>> node_transform = FuncNodeTransform(custom_node_transform, trans_node=True)

    # Example 3: Using with Document
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="custom", transform=text_transform)
    """
    def __init__(self, func: Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]],
                 trans_node: bool = None, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self._func, self._trans_node = func, trans_node
        self._need_ref = 'ref' in inspect.signature(func).parameters

    def transform(self, node: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        """
使用包装的用户自定义函数转换文档节点。

此方法将用户自定义函数应用于节点的文本内容（当 trans_node=False 时）或节点本身（当 trans_node=True 时）。

Args:
    node (DocNode): 要转换的文档节点。
    **kwargs: 传递给转换函数的额外关键字参数。

**Returns:**

- List[Union[str, DocNode]]: 转换结果，根据函数实现可以是字符串或 DocNode 对象。
"""
        if ref := kwargs.pop('ref', None):
            assert self._need_ref, \
                'if node group has ref, the transform function must support ref parameter.'
            kwargs['ref'] = ref if self._trans_node else [r.get_text() for r in ref]
        result = self._func(node if self._trans_node else node.get_text(), **kwargs)
        return result if isinstance(result, list) else [result]


class LLMParser(NodeTransform):
    """
一个文本摘要和关键词提取器，负责分析用户输入的文本，并根据请求任务提供简洁的摘要或提取相关关键词。

Args:
    llm (TrainableModule): 可训练的模块
    language (str): 语言种类，目前只支持中文（zh）和英文（en）
    task_type (str): 目前支持两种任务：摘要（summary）和关键词抽取（keywords）。
    num_workers (int): 控制并行处理的线程/进程数量。


Examples:

    >>> from lazyllm import TrainableModule
    >>> from lazyllm.tools.rag import LLMParser
    >>> llm = TrainableModule("internlm2-chat-7b")
    >>> summary_parser = LLMParser(llm, language="en", task_type="summary")
    """
    supported_languages = {'en': 'English', 'zh': 'Chinese'}

    def __init__(self, llm: TrainableModule, language: str, task_type: str,
                 prompts: Optional[LLMTransformParserPrompts] = None, num_workers: int = 30):
        super(__class__, self).__init__(num_workers=num_workers)
        assert language in self.supported_languages, f'Not supported language {language}'
        assert task_type in ['summary', 'keywords', 'qa', 'qa_img'], f'Not supported task_type {task_type}'
        self._task_type = task_type
        self._prompts = prompts or LLMTransformParserPrompts()
        task_prompt_tempalte = getattr(self._prompts, self._task_type)
        task_prompt = task_prompt_tempalte.format(language=self.supported_languages[language])
        if self._task_type == 'qa_img':
            prompt = dict(system=task_prompt, user='{input}')
        else:
            prompt = dict(system=task_prompt, user='#input:\n{input}\n#output:\n')
        self._llm = llm.share(prompt=AlpacaPrompter(prompt), stream=False, format=self._format)
        self._task_type = task_type

    def transform(self, node: DocNode, **kwargs) -> List[str]:
        """
在指定的文档上执行设定的任务。

Args:
    node (DocNode): 需要执行抽取任务的文档。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import LLMParser
    >>> llm = lazyllm.TrainableModule("internlm2-chat-7b").start()
    >>> m = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
    >>> summary_parser = LLMParser(llm, language="en", task_type="summary")
    >>> keywords_parser = LLMParser(llm, language="en", task_type="keywords")
    >>> documents = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
    >>> rm = lazyllm.Retriever(documents, group_name='CoarseChunk', similarity='bm25', topk=6)
    >>> doc_nodes = rm("test")
    >>> summary_result = summary_parser.transform(doc_nodes[0])
    >>> keywords_result = keywords_parser.transform(doc_nodes[0])
    """
        if self._task_type == 'qa_img':
            inputs = encode_query_with_filepaths('Extract QA pairs from images.', [node.image_path])
        else:
            inputs = node.get_text()
        result = self._llm(inputs)
        return [result] if isinstance(result, str) else result

    def _format(self, input):
        if isinstance(input, dict):
            input = input.get('output', input.get('text', input.get('content', str(input))))

        if not isinstance(input, str):
            input = str(input)

        if self._task_type == 'keywords':
            return [s.strip() for s in input.split(',')]
        elif self._task_type in ('qa', 'qa_img'):
            return [QADocNode(query=q.strip()[3:].strip(), answer=a.strip()[3:].strip()) for q, a in zip(
                list(filter(None, map(str.strip, input.split('\n'))))[::2],
                list(filter(None, map(str.strip, input.split('\n'))))[1::2])]
        return input

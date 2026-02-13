from typing import Optional, Dict, Any, Union, Callable, List
from enum import Enum, auto
from collections import defaultdict
from lazyllm.thirdparty import PIL
from lazyllm import JsonFormatter, config, reset_on_pickle, Mode, LOG
from lazyllm.components.utils.file_operate import _image_to_base64
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID
import uuid
import threading
import time
import hashlib
import copy
import json

_pickle_blacklist = {'_store', '_node_groups'}


class MetadataMode(str, Enum):
    """An enumeration."""
    ALL = auto()
    EMBED = auto()
    LLM = auto()
    NONE = auto()


@reset_on_pickle(('_lock', threading.Lock))
class DocNode:
    """
在指定的文档上执行设定的任务。

Args:
    uid(str): 唯一标识符。
    content(Union[str, List[Any]]):节点内容
    group(str):文档组名
    embedding(Dict[str, List[float]]):嵌入向量字典
    parent(Union[str, "DocNode"]):父节点引用
    store:存储表示
    node_groups(Dict[str, Dict]):节点存储组
    metadata(Dict[str, Any]):节点级元数据
    global_metadata(Dict[str, Any]):文档级元数据
    text(str):节点内容与content互斥
"""
    def __init__(self, uid: Optional[str] = None, content: Optional[Union[str, List[Any]]] = None,
                 group: Optional[str] = None, embedding: Optional[Dict[str, List[float]]] = None,
                 parent: Optional[Union[str, 'DocNode']] = None, store=None,
                 node_groups: Optional[Dict[str, Dict]] = None, metadata: Optional[Dict[str, Any]] = None,
                 global_metadata: Optional[Dict[str, Any]] = None, *, text: Optional[str] = None):
        if text and content:
            raise ValueError('`text` and `content` cannot be set at the same time.')
        if not content and not text: content = ''
        self._uid: str = uid if uid else str(uuid.uuid4())
        self._content: Optional[Union[str, List[Any]]] = content if content is not None else text
        self._group: Optional[str] = group
        self._embedding: Optional[Dict[str, List[float]]] = embedding or {}
        # metadata: the chunk's meta
        self._metadata: Dict[str, Any] = metadata or {}
        # Global metadata: the file's global metadata (higher level)
        self._global_metadata = global_metadata or {}
        # Metadata keys that are excluded from text for the embed model.
        self._excluded_embed_metadata_keys: List[str] = []
        # Metadata keys that are excluded from text for the LLM.
        self._excluded_llm_metadata_keys: List[str] = []
        # NOTE: node in parent should be id when stored in db (use store to recover): parent: 'uid'
        self._parent: Optional[Union[str, 'DocNode']] = parent
        self._children: Dict[str, List['DocNode']] = defaultdict(list)
        self._children_loaded = False
        self._store = store
        self._node_groups: Dict[str, Dict] = node_groups or {}
        self._lock = threading.Lock()
        self._embedding_state = set()
        self.relevance_score = None
        self.similarity_score = None
        self._content_hash: Optional[str] = None

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def group(self) -> str:
        return self._group

    @property
    def content(self) -> Union[str, List[Any]]:
        return self._content

    @content.setter
    def content(self, value: Union[str, List[Any]]) -> None:
        self._content = value
        self._content_hash = None

    @property
    def number(self) -> int:
        return self._metadata.get('lazyllm_store_num', 0)

    @number.setter
    def number(self, value: int) -> None:
        self._metadata['lazyllm_store_num'] = value

    @property
    def text(self) -> str:
        if isinstance(self._content, str):
            return self._content
        elif isinstance(self._content, list):
            if unexcepted := set([type(ele) for ele in self._content if not isinstance(ele, str)]):
                raise TypeError(f'Found non-string element in content: {unexcepted}')
            return '\n'.join(self._content)
        else:
            raise TypeError(f'content type "{type(self._content)}" is neither a str nor a list')

    @property
    def content_hash(self) -> str:
        if self._content_hash is None:
            self._content_hash = hashlib.sha256(self.text.encode('utf-8')).hexdigest()
        return self._content_hash

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, v: Optional[Dict[str, List[float]]]):
        self._embedding = v

    def _load_from_store(self, group_name: str, uids: Union[str, List[str]]) -> List['DocNode']:
        if not self._store or not uids:
            return []
        if isinstance(uids, str):
            uids = [uids]
        nodes = self._store.get_nodes(group=group_name, uids=uids,
                                      kb_id=self.global_metadata.get(RAG_KB_ID), display=True)
        for n in nodes:
            n._store = self._store
            n._node_groups = self._node_groups
        return nodes

    @property
    def parent(self) -> Optional['DocNode']:
        if self._parent and isinstance(self._parent, str) and self._node_groups:
            parent_group = self._node_groups[self._group]['parent']
            loaded = self._load_from_store(parent_group, self._parent)
            self._parent = loaded[0] if loaded else None
        return self._parent

    @parent.setter
    def parent(self, v: Optional['DocNode']):
        self._parent = v

    @property
    def children(self) -> Dict[str, List['DocNode']]:
        if not self._children_loaded and self._store and self._node_groups:
            self._children_loaded = True
            kb_id = self.global_metadata.get(RAG_KB_ID)
            doc_id = self.global_metadata.get(RAG_DOC_ID)
            c_groups = [grp for grp in self._node_groups.keys() if self._node_groups[grp]['parent'] == self._group]
            for grp in c_groups:
                if not self._store.is_group_active(grp):
                    continue
                nodes = self._store.get_nodes(group=grp, kb_id=kb_id, doc_ids=[doc_id])
                c_nodes = [n for n in nodes if n._parent in {self, self._uid}]
                self._children[grp] = c_nodes
                for n in self._children[grp]:
                    n._store = self._store
                    n._node_groups = self._node_groups
        return self._children

    @children.setter
    def children(self, v: Dict[str, List['DocNode']]):
        self._children = v

    @property
    def root_node(self) -> 'DocNode':
        node = self
        while isinstance(node._parent, DocNode):
            node = node._parent
        return node

    @property
    def is_root_node(self) -> bool:
        return (not self.parent)

    @property
    def global_metadata(self) -> Dict[str, Any]:
        return self.root_node._global_metadata

    @global_metadata.setter
    def global_metadata(self, global_metadata: Dict) -> None:
        self._global_metadata = global_metadata

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        self._metadata = metadata

    @property
    def excluded_embed_metadata_keys(self) -> List:
        return list(set(self.root_node._excluded_embed_metadata_keys + self._excluded_embed_metadata_keys))

    @excluded_embed_metadata_keys.setter
    def excluded_embed_metadata_keys(self, excluded_embed_metadata_keys: List) -> None:
        self._excluded_embed_metadata_keys = excluded_embed_metadata_keys

    @property
    def excluded_llm_metadata_keys(self) -> List:
        return list(set(self.root_node._excluded_llm_metadata_keys + self._excluded_llm_metadata_keys))

    @excluded_llm_metadata_keys.setter
    def excluded_llm_metadata_keys(self, excluded_llm_metadata_keys: List) -> None:
        self._excluded_llm_metadata_keys = excluded_llm_metadata_keys

    @property
    def docpath(self) -> str:
        return self.root_node.global_metadata.get(RAG_DOC_PATH, '')

    @docpath.setter
    def docpath(self, path):
        assert not self.parent, 'Only root node can set docpath'
        self.global_metadata[RAG_DOC_PATH] = str(path)

    def get_children_str(self) -> str:
        """获取子节点的字符串表示。

**Returns:**

- str: 返回一个字符串，表示子节点的字典格式，其中键为组名，值为该组中所有子节点的UID列表。
"""
        return str(
            {key: [node._uid for node in nodes] for key, nodes in self.children.items()}
        )

    def get_parent_id(self) -> str:
        """获取父节点的唯一标识符。

**Returns:**

- str: 返回父节点的UID，如果没有父节点则返回空字符串。
"""
        return self.parent._uid if self.parent else ''

    def __str__(self) -> str:
        return (
            f'DocNode(id: {self._uid}, group: {self._group}, content: {self._content}) parent: {self.get_parent_id()}, '
            f'children: {self.get_children_str()}'
        )

    def __repr__(self) -> str:
        return str(self) if config['mode'] == Mode.Debug else f'<Node id={self._uid}>'

    def __eq__(self, other):
        if isinstance(other, DocNode):
            return self._uid == other._uid
        return False

    def __hash__(self):
        return hash(self._uid)

    def __getstate__(self):
        st = self.__dict__.copy()
        for attr in _pickle_blacklist:
            st[attr] = None
        return st

    def has_missing_embedding(self, embed_keys: Union[str, List[str]]) -> List[str]:
        """
检查缺失的嵌入向量

Args:
    embed_keys(Union[str, List[str]]): 目标键列表
"""
        if isinstance(embed_keys, str): embed_keys = [embed_keys]
        assert len(embed_keys) > 0, 'The ebmed_keys to be checked must be passed in.'
        if self.embedding is None: return embed_keys
        return [k for k in embed_keys if k not in self.embedding]

    def do_embedding(self, embed: Dict[str, Callable]) -> None:
        """
执行嵌入计算

Args:
    embed(Dict[str, Callable]): 目标嵌入对象
"""
        generate_embed = {k: e(self.get_text(MetadataMode.EMBED)) for k, e in embed.items()}
        with self._lock:
            self.embedding = self.embedding or {}
            self.embedding = {**self.embedding, **generate_embed}

    def set_embedding(self, embed_key, embed_value) -> None:
        """设置文档节点的嵌入向量。

为文档节点设置指定键的嵌入向量值，用于后续的检索和相似度计算。

Args:
    embed_key (str): 嵌入向量的键名
    embed_value: 嵌入向量的值

Returns:
    None
"""
        with self._lock:
            self.embedding = self.embedding or {}
            self.embedding[embed_key] = embed_value

    def check_embedding_state(self, embed_key: str) -> None:
        """
阻塞检查嵌入状态,确保异步嵌入计算完成

Args:
    embed_key(str): 目标键列表
"""
        while True:
            with self._lock:
                if not self.has_missing_embedding(embed_key):
                    self._embedding_state.discard(embed_key)
                    break
            time.sleep(1)

    def get_content(self) -> str:
        """获取节点的内容文本，包含LLM模式的元数据。

**Returns:**

- str: 返回节点的文本内容，包含根据LLM模式格式化的元数据信息。
"""
        return self.get_text(MetadataMode.LLM)

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """
获取格式化元数据字符串

Args:
    mode: MetadataMode.NONE返回空字符串；
          MetadataMode.LLM过滤排除LLM不需要的元数据；
          MetadataMode.EMBED过滤排除嵌入模型不需要的元数据；
          MetadataMode.ALL返回全部元数据。
"""
        if mode == MetadataMode.NONE:
            return ''

        metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in metadata_keys:
                    metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in metadata_keys:
                    metadata_keys.remove(key)

        return '\n'.join([f'{key}: {self.metadata[key]}' for key in metadata_keys])

    def get_text(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """
组合元数据和内容

Args:
    metadata_mode: 与get_metadata_str中参数一致
"""
        metadata_str = self.get_metadata_str(metadata_mode).strip()
        if not metadata_str:
            return self.text if self.text else ''
        return f'{metadata_str}\n\n{self.text}'.strip()

    def to_dict(self) -> Dict:
        """
转换为字典格式
"""
        return dict(content=self._content, embedding=self.embedding, metadata=self.metadata)

    def with_score(self, score):
        """
浅拷贝原节点并添加语义相关分数。

Args:
    score: 相关性得分
"""
        node = copy.copy(self)
        node.relevance_score = score
        return node

    def with_sim_score(self, score):
        """
浅拷贝原节点并添加相似度分数。

Args:
    score: 相似度得分
"""
        node = copy.copy(self)
        node.similarity_score = score
        return node


class QADocNode(DocNode):
    """问答文档节点类，用于存储问答对数据。

Args:
    query (str): 问题文本。
    answer (str): 答案文本。
    uid (str): 唯一标识符。
    group (str): 文档组名。
    embedding (Dict[str, List[float]]): 嵌入向量字典。
    parent (DocNode): 父节点引用。
    metadata (Dict[str, Any]): 节点级元数据。
    global_metadata (Dict[str, Any]): 文档级元数据。
    text (str): 节点内容，与query互斥。
"""
    def __init__(self, query: str, answer: str, uid: Optional[str] = None, group: Optional[str] = None,
                 embedding: Optional[Dict[str, List[float]]] = None, parent: Optional['DocNode'] = None,
                 metadata: Optional[Dict[str, Any]] = None, global_metadata: Optional[Dict[str, Any]] = None,
                 *, text: Optional[str] = None):
        super().__init__(uid, query, group, embedding, parent, metadata=metadata,
                         global_metadata=global_metadata, text=text)
        self._answer = answer.strip()

    @property
    def answer(self) -> str:
        return self._answer

    def get_text(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """获取节点的文本内容。

Args:
    metadata_mode (MetadataMode): 元数据模式，默认为MetadataMode.NONE。
        当设置为MetadataMode.LLM时，返回格式化的问答对。
        其他模式下返回基类的文本格式。

**Returns:**

- str: 格式化后的文本内容。
"""
        if metadata_mode == MetadataMode.LLM:
            return f'query:\n{self.text}\nanswer\n{self._answer}'
        return super().get_text(metadata_mode)


class ImageDocNode(DocNode):
    """专门用于处理RAG系统中图像内容的文档节点。

ImageDocNode继承自DocNode，为图像处理和嵌入生成提供专门的功能。它自动处理图像加载、用于嵌入的base64编码，以及用于LLM处理的PIL图像对象。

Args:
    image_path (str): 图像文件的文件路径。这应该是一个有效的图像文件路径（例如.jpg、.png、.jpeg）。
    uid (Optional[str]): 文档节点的唯一标识符。如果未提供，将自动生成UUID。
    group (Optional[str]): 此节点所属的组名。用于组织和过滤节点。
    embedding (Optional[Dict[str, List[float]]]): 图像的预计算嵌入。键是嵌入模型名称，值是嵌入向量。
    parent (Optional[DocNode]): 文档层次结构中的父节点。用于构建文档树。
    metadata (Optional[Dict[str, Any]]): 与图像节点关联的附加元数据。
    global_metadata (Optional[Dict[str, Any]]): 适用于文档中所有节点的全局元数据。
    text (Optional[str]): 图像的可选文本描述或标题。


Examples:
    >>> from lazyllm.tools.rag.doc_node import ImageDocNode, MetadataMode
    >>> import numpy as np
    >>> image_node = ImageDocNode(
    ...     image_path="/home/mnt/yehongfei/Code/Test/framework.jpg",
    ...     text="这是一张照片"
    )
    >>> def clip_emb(content, modality="image"):
    ...     if modality == "image":
    ...         return [np.random.rand(512).tolist()]
    ...     return [np.random.rand(256).tolist()]
    >>> embed_functions = {"clip": clip_emb}
    >>> image_node.do_embedding(embed_functions)
    >>> print(f"嵌入维度: {len(image_node.embedding['clip'])}")
    >>> text_representation = image_node.get_text()
    >>> content_representation = image_node.get_content(MetadataMode.EMBED)
    >>> print(f"text属性: {text_representation}")
    >>> print(f"content属性: {content_representation}")
    """
    def __init__(self, image_path: str, uid: Optional[str] = None, group: Optional[str] = None,
                 embedding: Optional[Dict[str, List[float]]] = None, parent: Optional['DocNode'] = None,
                 metadata: Optional[Dict[str, Any]] = None, global_metadata: Optional[Dict[str, Any]] = None,
                 *, text: Optional[str] = None):
        super().__init__(uid, None, group, embedding, parent, metadata=metadata,
                         global_metadata=global_metadata, text=text)
        self._image_path = image_path.strip()
        self._modality = 'image'

    def do_embedding(self, embed: Dict[str, Callable]) -> None:
        """使用提供的嵌入函数为图像生成嵌入。

此方法重写父类方法以处理图像特定的嵌入生成。它自动将图像转换为适当的格式（用于嵌入的base64），并使用图像模态调用嵌入函数。

Args:
    embed (Dict[str, Callable]): 嵌入函数字典。键是嵌入模型名称，值是接受(content, modality)并返回嵌入向量的可调用函数。
"""
        for k, e in embed.items():
            emb = e(self.get_content(MetadataMode.EMBED), modality=self._modality)
            generate_embed = {k: emb[0]}

        with self._lock:
            self.embedding = self.embedding or {}
            self.embedding = {**self.embedding, **generate_embed}

    def get_content(self, metadata_mode=MetadataMode.LLM) -> str:
        """根据元数据模式获取不同格式的图像内容。

此方法根据预期用例返回不同格式的图像内容。对于LLM处理，它返回PIL图像对象。对于嵌入生成，它返回base64编码的图像字符串。

Args:
    metadata_mode (MetadataMode, optional): 内容检索模式。默认为MetadataMode.LLM。
        - MetadataMode.LLM: 返回用于LLM处理的PIL图像对象
        - MetadataMode.EMBED: 返回用于嵌入生成的base64编码图像
        - 其他模式: 返回图像路径作为文本

**Returns:**

- Union[PIL.Image.Image, List[str], str]: 请求格式的图像内容。
"""
        if metadata_mode == MetadataMode.LLM:
            return PIL.Image.open(self._image_path)
        elif metadata_mode == MetadataMode.EMBED:
            image_base64, mime = _image_to_base64(self._image_path)
            return [f'data:{mime};base64,{image_base64}']
        else:
            return self.get_text()

    @property
    def image_path(self):
        return self._image_path

    def get_text(self) -> str:  # Disable access to self._content
        """获取图像路径作为文本表示。

此方法重写父类方法以返回图像路径而不是内容字段，因为ImageDocNode不使用内容字段存储文本。

**Returns:**

- str: 图像文件路径。
"""
        return self._image_path

    @property
    def text(self) -> str:  # Disable access to self._content
        return self._image_path

class JsonDocNode(DocNode):
    """用于处理RAG系统中JSON内容的专用文档节点。

JsonDocNode继承自DocNode，提供存储和处理JSON数据（字典或列表）的功能。它自动将JSON内容序列化为字符串格式，并支持通过JsonFormatter进行自定义格式化。

Args:
    uid (Optional[str]): 文档节点的唯一标识符。如果未提供，将自动生成UUID。
    content (Optional[Union[Dict[str, Any], List[Any]]]): 要存储的JSON内容。可以是字典或列表。
    group (Optional[str]): 此节点所属的组名。用于组织和过滤节点。
    embedding (Optional[Dict[str, List[float]]]): 预计算的嵌入。键是嵌入模型名称，值是嵌入向量。
    parent (Optional[DocNode]): 文档层次结构中的父节点。用于构建文档树。
    metadata (Optional[Dict[str, Any]]): 与节点关联的附加元数据。
    global_metadata (Optional[Dict[str, Any]]): 适用于文档中所有节点的全局元数据。
    formatter (JsonFormatter, optional): 用于自定义JSON内容表示的格式化器。在获取用于嵌入的内容时使用。

Notes:
    - text属性返回序列化为字符串的JSON内容。
    - 当提供formatter时，get_content()在向量化模式下使用它进行输出格式化，仅向量化指定的字段，使用换行符连接。
"""
    def __init__(self, uid: Optional[str] = None, content: Optional[Union[Dict[str, Any], List[Any]]] = None,
                 group: Optional[str] = None, embedding: Optional[Dict[str, List[float]]] = None,
                 parent: Optional['DocNode'] = None, metadata: Optional[Dict[str, Any]] = None,
                 global_metadata: Optional[Dict[str, Any]] = None, *, formatter_str: Optional[str] = None):
        super().__init__(uid, content, group, embedding, parent, metadata=metadata, global_metadata=global_metadata)
        if formatter_str is not None:
            self.metadata['formatter_str'] = formatter_str
        else:
            formatter_str = self.metadata.get('formatter_str', '')
        self._formatter = JsonFormatter(formatter_str)

    @property
    def text(self) -> str:
        try:
            return json.dumps(self._content, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f'Cannot convert content to JSON string: {e}')

    @property
    def json_object(self) -> Union[Dict[str, Any], List[Any]]:
        return self._content

    def get_content(self, metadata_mode=MetadataMode.EMBED) -> str:
        if metadata_mode == MetadataMode.EMBED:
            try:
                return json.dumps(self._formatter(self._content), ensure_ascii=False)
            except (TypeError, ValueError) as e:
                LOG.warning(f'Cannot convert content to JSON string: {e}')
        return self.text

    def _serialize_content(self) -> str:
        return self.text

    @staticmethod
    def _deserialize_content(content: str) -> Union[Dict[str, Any], List[Any]]:
        return json.loads(content)

class RichDocNode(DocNode):
    """用于聚合多个带有独立元数据的段落节点的专用文档节点，以保持每个文档进有一个root node。

RichDocNode继承自DocNode，用于封装reader返回的多个子节点（通常是段落）。它保留完整的文档文本内容，同时允许每个子节点维护自己的元数据。结合RichTransform使用时，可以恢复出原始的DocNode实例（带有元信息）。

Args:
    nodes (List[DocNode]): 要聚合的段落节点列表。每个节点的文本会被合并到content中。
    uid (Optional[str]): 文档节点的唯一标识符。如果未提供，将自动生成UUID。
    group (Optional[str]): 此节点所属的组名。用于组织和过滤节点。
    embedding (Optional[Dict[str, List[float]]]): 预计算的嵌入。键是嵌入模型名称，值是嵌入向量。
    parent (Optional[DocNode]): 文档层次结构中的父节点。用于构建文档树。
    metadata (Optional[Dict[str, Any]]): 与节点关联的附加元数据。
    global_metadata (Optional[Dict[str, Any]]): 适用于文档中所有节点的全局元数据。

Notes:
    - 通常由PDF reader在单个文档产生多个节点时返回，作为root node。
    - 原始段落节点存储在内部，可通过RichTransform访问恢复。
    - 以段落文本列表的形式在content字段中保留整篇文档文本。
"""
    def __init__(self, nodes: List[DocNode], uid: Optional[str] = None,
                 group: Optional[str] = None, embedding: Optional[Dict[str, List[float]]] = None,
                 parent: Optional['DocNode'] = None, metadata: Optional[Dict[str, Any]] = None,
                 global_metadata: Optional[Dict[str, Any]] = None):
        super().__init__(uid, [n.text for n in nodes], group, embedding, parent, metadata=metadata,
                         global_metadata=global_metadata)
        self._nodes: List[DocNode] = nodes

    @property
    def nodes(self) -> List[DocNode]:
        return self._nodes

    def _serialize_nodes(self) -> str:

        def _serialize_node(node: DocNode) -> str:
            formatted_node = {
                'content': node.text,
                'metadata': node.metadata,
                'global_metadata': node.global_metadata,
                'excluded_embed_metadata_keys': node.excluded_embed_metadata_keys,
                'excluded_llm_metadata_keys': node.excluded_llm_metadata_keys,
            }
            return json.dumps(formatted_node, ensure_ascii=False)

        return json.dumps([_serialize_node(n) for n in self.nodes])

    @staticmethod
    def _deserialize_nodes(nodes_content: str) -> List[DocNode]:

        def _deserialize_node(content: str) -> DocNode:
            formatted_node = json.loads(content)
            node = DocNode(content=formatted_node['content'], metadata=formatted_node['metadata'],
                           global_metadata=formatted_node['global_metadata'])
            node.excluded_embed_metadata_keys = formatted_node['excluded_embed_metadata_keys']
            node.excluded_llm_metadata_keys = formatted_node['excluded_llm_metadata_keys']
            return node

        return [_deserialize_node(content) for content in json.loads(nodes_content)]

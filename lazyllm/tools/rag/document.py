import os
from typing import Callable, Optional, Dict, Union, List, Type, Set
from functools import cached_property
from pydantic import BaseModel
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated, OnlineChatModule, TrainableModule
from lazyllm.module import LLMBase
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from lazyllm.common.bind import _MetaBind

from .doc_manager import DocManager
from .doc_impl import DocImpl, StorePlaceholder, EmbedPlaceholder, BuiltinGroups, DocumentProcessor, NodeGroupType
from .doc_node import DocNode
from .doc_to_db import DocInfoSchema, DocToDbProcessor, extract_db_schema_from_files, SchemaExtractor
from .store import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY
from .store.store_base import DEFAULT_KB_ID
from .index_base import IndexBase
from .utils import DocListManager, ensure_call_endpoint
from .global_metadata import GlobalMetadataDesc as DocField
from .web import DocWebModule
import copy
import functools
import weakref


class CallableDict(dict):
    def __call__(self, cls, *args, **kw):
        return self[cls](*args, **kw)


class _MetaDocument(_MetaBind):
    def __instancecheck__(self, __instance):
        if isinstance(__instance, UrlDocument): return True
        return super().__instancecheck__(__instance)


class Document(ModuleBase, BuiltinGroups, metaclass=_MetaDocument):
    """初始化一个文档管理模块，支持可选的向量化、存储和用户界面。

``Document`` 模块提供了统一的文档数据集管理接口，支持本地文件、云端文件或临时文档文件。它可以选择运行文档管理服务或 Web UI，并支持多种向量化模型和自定义存储后端。

Args:
    dataset_path (Optional[str]): 数据集目录路径。如果路径不存在，系统会尝试在 ``lazyllm.config["data_path"]`` 中查找。
    embed (Optional[Union[Callable, Dict[str, Callable]]]): 文档向量化函数或函数字典。若为字典，键为 embedding 名称，值为对应的模型。
    manager (Union[bool, str], optional): 是否启用文档管理服务。``True`` 表示启动管理服务；``'ui'`` 表示同时启动 Web 管理界面；默认 ``False``。
    server (Union[bool, int], optional): 是否为知识库运行服务接口。``True`` 表示启动默认服务；整型数值表示自定义端口；``False`` 表示关闭。默认为 ``False``。
    name (Optional[str]): 文档集合的名称标识符。默认为系统默认名称。
    launcher (Optional[Launcher]): 启动器实例，用于管理服务进程。默认使用远程异步启动器。
    store_conf (Optional[Dict]): 存储配置。默认使用内存中的 MapStore。
    doc_fields (Optional[Dict[str, DocField]]): 元数据字段配置，用于存储和检索文档属性。
    cloud (bool): 是否为云端数据集。默认为 ``False``。
    doc_files (Optional[List[str]]): 临时文档文件列表。当使用此参数时，``dataset_path`` 必须为 ``None``，且仅支持 MapStore。
    processor (Optional[DocumentProcessor]): 文档处理服务。
    display_name (Optional[str]): 文档模块的可读显示名称。默认为集合名称。
    description (Optional[str]): 文档集合的描述。默认为 ``"algorithm description"``。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import Document
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)  # or documents = Document(dataset_path='your_doc_path', embed={"key": m}, manager=False)
    >>> m1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
    >>> document1 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, manager=False)

    >>> store_conf = {
    >>>     "segment_store": {
    >>>         "type": "map",
    >>>         "kwargs": {
    >>>             "uri": "/tmp/tmp_segments.db",
    >>>         },
    >>>     },
    >>>     "vector_store": {
    >>>         "type": "milvus",
    >>>         "kwargs": {
    >>>             "uri": "/tmp/tmp_milvus.db",
    >>>             "index_kwargs": {
    >>>                 "index_type": "FLAT",
    >>>                 "metric_type": "COSINE",
    >>>             },
    >>>         },
    >>>     },
    >>> }
    >>> doc_fields = {
    >>>     'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
    >>>     'public_year': DocField(data_type=DataType.INT32),
    >>> }
    >>> document2 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, store_conf=store_conf, doc_fields=doc_fields)
    """
    class _Manager(ModuleBase):
        def __init__(self, dataset_path: Optional[str], embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: Union[bool, str] = False, server: Union[bool, int] = False, name: Optional[str] = None,
                     launcher: Optional[Launcher] = None, store_conf: Optional[Dict] = None,
                     doc_fields: Optional[Dict[str, DocField]] = None, cloud: bool = False,
                     doc_files: Optional[List[str]] = None, processor: Optional[DocumentProcessor] = None,
                     display_name: Optional[str] = '', description: Optional[str] = 'algorithm description',
                     schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None):
            super().__init__()
            self._origin_path, self._doc_files, self._cloud = dataset_path, doc_files, cloud

            if dataset_path and not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config['data_path'], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            elif dataset_path:
                dataset_path = os.path.join(os.getcwd(), dataset_path)

            self._launcher: Launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = self._get_embeds(embed)
            self._processor = processor
            name = name or DocListManager.DEFAULT_GROUP_NAME
            if not display_name: display_name = name

            self._dlm = None if (self._cloud or self._doc_files is not None) else DocListManager(
                dataset_path, name, enable_path_monitoring=False if manager else True)
            self._kbs = CallableDict({name: DocImpl(
                embed=self._embed, dlm=self._dlm, doc_files=doc_files, global_metadata_desc=doc_fields,
                store=store_conf, processor=processor, algo_name=name, display_name=display_name,
                description=description, schema_extractor=schema_extractor)})

            if manager: self._manager = ServerModule(DocManager(self._dlm), launcher=self._launcher)
            if manager == 'ui': self._docweb = DocWebModule(doc_server=self._manager)
            if server: self._kbs = ServerModule(self._kbs, port=(None if isinstance(server, bool) else int(server)))
            self._global_metadata_desc = doc_fields

        @property
        def url(self):
            if hasattr(self, '_manager'): return self._manager._url
            return None

        @property
        @deprecated('Document.manager.url')
        def _url(self):
            return self.url

        @property
        def web_url(self):
            if hasattr(self, '_docweb'): return self._docweb.url
            return None

        def _get_embeds(self, embed):
            embeds = embed if isinstance(embed, dict) else {EMBED_DEFAULT_KEY: embed} if embed else {}
            for embed in embeds.values():
                if isinstance(embed, ModuleBase):
                    self._submodules.append(embed)
            return embeds

        def add_kb_group(self, name, doc_fields: Optional[Dict[str, DocField]] = None,
                         store_conf: Optional[Dict] = None,
                         embed: Optional[Union[Callable, Dict[str, Callable]]] = None):
            embed = self._get_embeds(embed) if embed else self._embed
            if isinstance(self._kbs, ServerModule):
                self._kbs._impl._m[name] = DocImpl(dlm=self._dlm, embed=embed, kb_group_name=name,
                                                   global_metadata_desc=doc_fields, store=store_conf)
            else:
                self._kbs[name] = DocImpl(dlm=self._dlm, embed=self._embed, kb_group_name=name,
                                          global_metadata_desc=doc_fields, store=store_conf)
            self._dlm.add_kb_group(name=name)

        def get_doc_by_kb_group(self, name):
            return self._kbs._impl._m[name] if isinstance(self._kbs, ServerModule) else self._kbs[name]

        def stop(self):
            if hasattr(self, '_docweb'):
                self._docweb.stop()
            self._launcher.cleanup()

        def __call__(self, *args, **kw):
            return self._kbs(*args, **kw)

    def __new__(cls, *args, **kw):
        if url := kw.pop('url', None):
            name = kw.pop('name', None)
            if args or kw:
                raise TypeError(
                    f"When 'url' is provided, only 'name' is allowed. "
                    f'Got args={args}, extra kwargs={kw}'
                )
            return UrlDocument(url, name)
        else:
            return super().__new__(cls)

    def __init__(self, dataset_path: Optional[str] = None, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = False, manager: Union[bool, str, 'Document._Manager', DocumentProcessor] = False,
                 server: Union[bool, int] = False, name: Optional[str] = None, launcher: Optional[Launcher] = None,
                 doc_files: Optional[List[str]] = None, doc_fields: Dict[str, DocField] = None,
                 store_conf: Optional[Dict] = None, display_name: Optional[str] = '',
                 description: Optional[str] = 'algorithm description',
                 schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None):
        super().__init__()
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
            manager = create_ui
        if isinstance(dataset_path, (tuple, list)):
            doc_fields = dataset_path
            dataset_path = None
        if doc_files is not None:
            assert dataset_path is None and not manager, (
                'Manager and dataset_path are not supported for Document with temp-files')
            assert store_conf is None or store_conf['type'] == 'map', (
                'Only map store is supported for Document with temp-files')

        name = name or DocListManager.DEFAULT_GROUP_NAME
        self._schema_extractor: SchemaExtractor = schema_extractor

        if isinstance(manager, Document._Manager):
            assert not server, 'Server infomation is already set to by manager'
            assert not launcher, 'Launcher infomation is already set to by manager'
            assert not manager._cloud, 'manager is not allowed to share in cloud mode'
            assert manager._doc_files is None, 'manager is not allowed to share with temp files'
            if dataset_path != manager._dataset_path and dataset_path != manager._origin_path:
                raise RuntimeError(f'Document path mismatch, expected `{manager._dataset_path}`'
                                   f'while received `{dataset_path}`')
            manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf, embed=embed)
            self._manager = manager
            self._curr_group = name
        else:
            if isinstance(manager, DocumentProcessor):
                processor, cloud = manager, True
                processor.start()
                manager = False
                assert name, '`Name` of Document is necessary when using cloud service'
                assert store_conf.get('type') != 'map', 'Cloud manager is not supported when using map store'
                assert not dataset_path, 'Cloud manager is not supported with local dataset path'
            else:
                cloud, processor = False, None
            self._manager = Document._Manager(dataset_path, embed, manager, server, name, launcher, store_conf,
                                              doc_fields, cloud=cloud, doc_files=doc_files, processor=processor,
                                              display_name=display_name, description=description,
                                              schema_extractor=self._schema_extractor)
            self._curr_group = name
        self._doc_to_db_processor: DocToDbProcessor = None
        self._graph_document: weakref.ref = None

    @staticmethod
    def list_all_files_in_directory(
        dataset_path: str, skip_hidden_path: bool = True, recursive: bool = True
    ) -> List[str]:
        """列出指定目录路径中的所有文件。

该方法会以递归或非递归方式遍历目录并收集所有文件路径。可以选择跳过隐藏文件和目录（以 “.” 开头的）。如果传入的路径本身是文件，则返回仅包含该文件路径的列表。

Args:
    dataset_path (str): 要列出文件列表的目录。
    skip_hidden_path (bool, optional): 是否跳过隐藏文件和目录（以 “.” 开头）。默认值为 True
    recursive (bool, optional): 是否递归搜索子目录。如果为 False，则只返回当前目录下的文件。默认值为 True。

**Returns:**

- List[str]: 绝对文件路径列表。如果路径不存在或不是目录，则返回空列表。
"""
        files_list = []

        if not os.path.exists(dataset_path):
            return files_list

        if not os.path.isdir(dataset_path):
            return [dataset_path] if os.path.isfile(dataset_path) else files_list

        if recursive:
            for root, dirs, files in os.walk(os.path.abspath(dataset_path)):
                # Skip hidden directories
                if skip_hidden_path:
                    path_parts = root.split(os.sep)
                    if any(part.startswith('.') for part in path_parts if part):
                        continue
                    # Filter out hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]

                # Skip hidden files
                if skip_hidden_path:
                    files = [file_path for file_path in files if not file_path.startswith('.')]

                files = [os.path.join(root, file_path) for file_path in files]
                files_list.extend(files)
        else:
            items = os.listdir(dataset_path)
            for item in items:
                item_path = os.path.join(dataset_path, item)
                # Skip hidden files/directories
                if skip_hidden_path and item.startswith('.'):
                    continue
                # Only add files, not directories
                if os.path.isfile(item_path):
                    files_list.append(item_path)

        return files_list

    def _list_all_files_in_dataset(self, skip_hidden_path: bool = True) -> List[str]:
        return self.list_all_files_in_directory(self._manager._dataset_path, skip_hidden_path)

    @property
    def url(self):
        assert isinstance(self._manager._kbs, ServerModule), 'Document is not a service, please set `manager` to `True`'
        return self._manager._kbs._url

    def connect_sql_manager(
        self,
        sql_manager: SqlManager,
        schma: Optional[DocInfoSchema] = None,
        force_refresh: bool = True,
    ):
        """连接 SQL 管理器并初始化文档与数据库的映射处理器。

此方法会验证数据库连接，并根据传入的文档表模式（schema）更新或重置数据库表结构。如果已存在的 schema 与新传入的 schema 不一致，则需要设置 ``force_refresh=True`` 以强制刷新。

Args:
    sql_manager (SqlManager): SQL 管理器实例，用于连接和操作数据库。
    schma (Optional[DocInfoSchema]): 文档表模式定义。包含字段名称、类型及描述。
    force_refresh (bool, optional): 当 schema 发生变化时，是否强制刷新数据库表结构。默认为 ``True``。

Raises:
    RuntimeError: 当数据库连接失败时抛出。
    AssertionError: 当未提供 schema 或 schema 变更时未设置 ``force_refresh`` 抛出。
"""
        def format_schema_to_dict(schema: DocInfoSchema):
            if schema is None:
                return None, None
            desc_dict = {ele['key']: ele['desc'] for ele in schema}
            type_dict = {ele['key']: ele['type'] for ele in schema}
            return desc_dict, type_dict

        def compare_schema(old_schema: DocInfoSchema, new_schema: DocInfoSchema):
            old_desc_dict, old_type_dict = format_schema_to_dict(old_schema)
            new_desc_dict, new_type_dict = format_schema_to_dict(new_schema)
            return old_desc_dict == new_desc_dict and old_type_dict == new_type_dict

        # 1. Check valid arguments
        if sql_manager.check_connection().status != DBStatus.SUCCESS:
            raise RuntimeError(f'Failed to connect to sql manager: {sql_manager._gen_conn_url()}')
        pre_doc_table_schema = None
        if self._doc_to_db_processor:
            pre_doc_table_schema = self._doc_to_db_processor.doc_info_schema
        assert pre_doc_table_schema or schma, 'doc_table_schma must be given'

        schema_equal = compare_schema(pre_doc_table_schema, schma)
        assert (
            schema_equal or force_refresh is True
        ), 'When changing doc_table_schema, force_refresh should be set to True'

        # 2. Init handler if needed
        need_init_processor = False
        if self._doc_to_db_processor is None:
            need_init_processor = True
        else:
            # avoid reinit for the same db
            if sql_manager != self._doc_to_db_processor.sql_manager:
                need_init_processor = True
        if need_init_processor:
            self._doc_to_db_processor = DocToDbProcessor(sql_manager)

        # 3. Reset doc_table_schema if needed
        if schma and not schema_equal:
            # This api call will clear existing db table 'lazyllm_doc_elements'
            self._doc_to_db_processor._reset_doc_info_schema(schma)

    def get_sql_manager(self):
        """获取当前文档模块绑定的 SQL 管理器实例。

**Returns:**

- SqlManager: 已连接的 SQL 管理器实例。
"""
        if self._doc_to_db_processor is None:
            raise ValueError('Please call connect_sql_manager to init handler first')
        return self._doc_to_db_processor.sql_manager

    def extract_db_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], print_schema: bool = False
    ) -> DocInfoSchema:
        """基于文档数据集和大语言模型自动提取数据库表模式（schema）。

此方法会扫描数据集中的所有文件，并调用大语言模型提取文档信息结构。可选择是否打印提取的 schema。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于解析文档并提取 schema 的模型。
    print_schema (bool, optional): 是否在日志中打印提取的 schema。默认为 ``False``。

**Returns:**

- DocInfoSchema: 提取的数据库表模式。
"""
        file_paths = self._list_all_files_in_dataset()
        schema = extract_db_schema_from_files(file_paths, llm)
        if print_schema:
            lazyllm.LOG.info(f'Extracted Schema:\n\t{schema}\n')
        return schema

    def update_database(self, llm: Union[OnlineChatModule, TrainableModule]):
        """使用大语言模型解析文档并将提取的信息更新到数据库。

此方法会遍历数据集中的所有文件，提取文档结构化信息，并将其写入数据库。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于解析文档并提取信息的大语言模型。
"""
        assert self._doc_to_db_processor, 'Please call connect_db to init handler first'
        file_paths = self._list_all_files_in_dataset()
        info_dicts = self._doc_to_db_processor.extract_info_from_docs(llm, file_paths)
        self._doc_to_db_processor.export_info_to_db(info_dicts)

    @deprecated('Document(dataset_path, manager=doc.manager, name=xx, doc_fields=xx, store_conf=xx)')
    def create_kb_group(self, name: str, doc_fields: Optional[Dict[str, DocField]] = None,
                        store_conf: Optional[Dict] = None) -> 'Document':
        """创建一个新的知识库分组（KB Group），并返回绑定到该分组的文档对象。

知识库分组用于在同一个文档模块中划分不同的文档集合，每个分组可以有独立的字段定义和存储配置。

Args:
    name (str): 知识库分组的名称。
    doc_fields (Optional[Dict[str, DocField]]): 文档字段定义。指定每个字段的名称、类型和描述。
    store_conf (Optional[Dict]): 存储配置，用于定义存储后端及其参数。

**Returns:**

- Document: 一个绑定到新建知识库分组的文档对象副本。
"""
        self._manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf)
        doc = copy.copy(self)
        doc._curr_group = name
        return doc

    @property
    @deprecated('Document._manager')
    def _impls(self): return self._manager

    @property
    def _impl(self) -> DocImpl: return self._manager.get_doc_by_kb_group(self._curr_group)

    @property
    def manager(self): return self._manager._processor or self._manager

    def activate_group(self, group_name: str, embed_keys: Optional[Union[str, List[str]]] = None,
                       enable_embed: bool = True):
        """激活指定的知识库分组，并可选择指定要启用的 embedding key。

激活后，文档模块会在该分组下执行检索和存储操作。如果未指定 embedding key，则默认启用所有可用的 embedding。

Args:
    group_name (str): 要激活的知识库分组名称。
    embed_keys (Optional[Union[str, List[str]]]): 需要启用的 embedding key，可以是单个字符串或字符串列表。默认为空列表，表示启用全部 embedding。
"""
        if embed_keys and not enable_embed:
            raise ValueError('`enable_embed` must be set to True when `embed_keys` is provided')
        # if embed_keys is None, use default embed keys
        if (enable_embed and not embed_keys) and self._manager._embed:
            embed_keys = self._manager._embed.keys()
        if isinstance(embed_keys, str): embed_keys = [embed_keys]
        self._impl.activate_group(group_name, embed_keys, enable_embed)

    def activate_groups(self, groups: Union[str, List[str]], **kwargs):
        """批量激活多个知识库分组。

该方法会依次调用 `activate_group` 来激活传入的所有分组。

Args:
    groups (Union[str, List[str]]): 要激活的分组名称或分组名称列表。
"""
        if isinstance(groups, str): groups = [groups]
        for group in groups:
            self.activate_group(group, **kwargs)

    @DynamicDescriptor
    def create_node_group(self, name: str = None, *, transform: Callable, parent: str = LAZY_ROOT_NAME,
                          trans_node: bool = None, num_workers: int = 0, display_name: str = None,
                          ref: str = None, group_type: NodeGroupType = NodeGroupType.CHUNK, **kwargs) -> None:
        """
创建一个由指定规则生成的 node group。

Args:
    name (str): node group 的名称。
    transform (Callable): 将 node 转换成 node group 的转换规则，函数原型是 `(DocNode, group_name, **kwargs) -> List[DocNode]`。目前内置的有 [SentenceSplitter][lazyllm.tools.SentenceSplitter]。用户也可以自定义转换规则。
    trans_node (bool): 决定了transform的输入和输出是 `DocNode` 还是 `str` ，默认为None。只有在 `transform` 为 `Callable` 时才可以设置为true。
    num_workers (int): Transform时所用的新线程数量，默认为0
    parent (str): 需要进一步转换的节点。转换之后得到的一系列新的节点将会作为该父节点的子节点。如果不指定则从根节点开始转换。
    ref (str): 当前节点组引用的其他节点组名称。引用的节点组必须是父节点组的后代。在转换时，ref 指定的节点组中的相关节点会作为参数传递给 transform 函数（如果 transform 函数支持 ref 参数）。
    kwargs: 和具体实现相关的参数。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    >>> # Example with ref parameter: create a node group that references another group
    >>> documents.create_node_group(name="fine_chunks", parent="sentences",
    ...                             transform=SentenceSplitter, chunk_size=128, chunk_overlap=12)
    >>> def transform_with_ref(text, ref):
    ...     # ref contains nodes from the referenced group
    ...     return "
    ".join(ref)
    >>> documents.create_node_group(name="summary_chunks", parent="sentences",
    ...                             transform=transform_with_ref, ref="fine_chunks")
    """
        assert ref is None or parent != ref, 'parent and ref must be different'
        if isinstance(self, type):
            DocImpl.create_global_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                             num_workers=num_workers, display_name=display_name,
                                             group_type=group_type, ref=ref, **kwargs)
        else:
            self._impl.create_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                         num_workers=num_workers, display_name=display_name, group_type=group_type,
                                         ref=ref, **kwargs)

    @DynamicDescriptor
    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        """
用于实例指定文件读取器，作用范围仅对注册的 Document 对象可见。注册的文件读取器必须是 Callable 对象。只能通过函数调用的方式进行注册。并且通过实例注册的文件读取器的优先级高于通过类注册的文件读取器，并且实例和类注册的文件读取器的优先级高于系统默认的文件读取器。即优先级的顺序是：实例文件读取器 > 类文件读取器 > 系统默认文件读取器。

Args:
    pattern (str): 文件读取器适用的匹配规则
    func (Callable): 文件读取器，必须是Callable的对象


Examples:

    >>> from lazyllm.tools.rag import Document, DocNode
    >>> from lazyllm.tools.rag.readers import ReaderBase
    >>> class YmlReader(ReaderBase):
    ...     def _load_data(self, file, fs=None):
    ...         try:
    ...             import yaml
    ...         except ImportError:
    ...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
    ...         with open(file, 'r') as f:
    ...             data = yaml.safe_load(f)
    ...         print("Call the class YmlReader.")
    ...         return [DocNode(text=data)]
    ...
    >>> def processYml(file):
    ...     with open(file, 'r') as f:
    ...         data = f.read()
    ...     print("Call the function processYml.")
    ...     return [DocNode(text=data)]
    ...
    >>> doc1 = Document(dataset_path="your_files_path", create_ui=False)
    >>> doc2 = Document(dataset_path="your_files_path", create_ui=False)
    >>> doc1.add_reader("**/*.yml", YmlReader)
    >>> print(doc1._impl._local_file_reader)
    {'**/*.yml': <class '__main__.YmlReader'>}
    >>> print(doc2._impl._local_file_reader)
    {}
    >>> files = ["your_yml_files"]
    >>> Document.register_global_reader("**/*.yml", processYml)
    >>> doc1._impl._reader.load_data(input_files=files)
    Call the class YmlReader.
    >>> doc2._impl._reader.load_data(input_files=files)
    Call the function processYml.
    """
        if isinstance(self, type):
            return DocImpl.register_global_reader(pattern=pattern, func=func)
        else:
            self._impl.add_reader(pattern, func)

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
        """
用于指定文件读取器，作用范围对于所有的 Document 对象都可见。注册的文件读取器必须是 Callable 对象。可以使用装饰器的方式进行注册，也可以通过函数调用的方式进行注册。

Args:
    pattern (str): 文件读取器适用的匹配规则
    func (Callable): 文件读取器，必须是Callable的对象


Examples:

    >>> from lazyllm.tools.rag import Document, DocNode
    >>> @Document.register_global_reader("**/*.yml")
    >>> def processYml(file):
    ...     with open(file, 'r') as f:
    ...         data = f.read()
    ...     return [DocNode(text=data)]
    ...
    >>> doc1 = Document(dataset_path="your_files_path", create_ui=False)
    >>> doc2 = Document(dataset_path="your_files_path", create_ui=False)
    >>> files = ["your_yml_files"]
    >>> docs1 = doc1._impl._reader.load_data(input_files=files)
    >>> docs2 = doc2._impl._reader.load_data(input_files=files)
    >>> print(docs1[0].text == docs2[0].text)
    # True
    """
        return cls.add_reader(pattern, func)

    def get_store(self):
        """获取存储占位符对象。

该方法返回一个存储层的占位符，用于延迟绑定具体的存储实现。调用者可以基于此对象进行存储相关的配置或扩展。

**Returns:**

- StorePlaceholder: 存储占位符对象。
"""
        return StorePlaceholder()

    def get_embed(self):
        """获取 embedding 占位符对象。

该方法返回一个 embedding 层的占位符，用于延迟绑定具体的 embedding 实现。调用者可以基于此对象进行 embedding 相关的配置或扩展。

**Returns:**

- EmbedPlaceholder: embedding 占位符对象。
"""
        return EmbedPlaceholder()

    def register_index(self, index_type: str, index_cls: IndexBase, *args, **kwargs) -> None:
        """注册索引类型。

该方法允许用户为文档模块注册新的索引类型，以便扩展检索能力。注册后，可以通过索引类型来调用对应的索引实现。

Args:
    index_type (str): 索引类型的名称。
    index_cls (IndexBase): 索引类，需继承自 ``IndexBase``。
    *args: 初始化索引类时的可变参数。
    **kwargs: 初始化索引类时的关键字参数。
"""
        self._impl.register_index(index_type, index_cls, *args, **kwargs)

    def _forward(self, func_name: str, *args, **kw):
        return self._manager(self._curr_group, func_name, *args, **kw)

    def find_parent(self, target) -> Callable:
        """查找目标的父节点。

该方法返回一个可调用对象，用于执行父节点查找操作。它会延迟调用底层实现以获取指定目标的父节点。

Args:
    target: 需要查找父节点的目标。

**Returns:**

- Callable: 可调用对象，用于执行父节点查找。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    >>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
    >>> documents.find_parent('children')
    """
        return functools.partial(self._forward, 'find_parent', group=target)

    def find_children(self, target) -> Callable:
        """查找目标的子节点。

该方法返回一个可调用对象，用于执行子节点查找操作。它会延迟调用底层实现以获取指定目标的所有子节点。

Args:
    target: 需要查找子节点的目标。

**Returns:**

- Callable: 可调用对象，用于执行子节点查找。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    >>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
    >>> documents.find_children('parent')
    """
        return functools.partial(self._forward, 'find_children', group=target)

    def find(self, target) -> Callable:
        """查找目标。

该方法返回一个可调用对象，用于执行目标查找操作。它会延迟调用底层实现以获取指定的目标对象。

Args:
    target: 需要查找的目标。

**Returns:**

- Callable: 可调用对象，用于执行目标查找。
"""
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw) -> List[DocNode]:
        return self._forward('retrieve', *args, **kw)

    def clear_cache(self, group_names: Optional[List[str]] = None) -> None:
        """清理缓存。

该方法用于清理文档模块的缓存，可以指定要清理的分组名称列表。如果未指定分组名称，则默认清理所有分组的缓存。

Args:
    group_names (Optional[List[str]]): 需要清理缓存的分组名称列表。默认为 ``None``，表示清理全部缓存。
"""
        return self._forward('clear_cache', group_names)

    def drop_algorithm(self):
        """
用于删除当前文档集合的在文档解析服务中注册的算法信息。
"""
        return self._forward('drop_algorithm')

    def analyze_schema_by_llm(self, kb_id: Optional[str] = None, doc_ids: Optional[List[str]] = None):
        """
用于使用大模型为文档管理模块中特定的知识库或文档集合自动抽取字段集合，返回自动生成的Pydantic Model。
支持传入特定知识库id和文档id列表。

Args:
    kb_id: 目标知识库id
    doc_ids: 目标文档id列表
"""
        return self._forward('_analyze_schema_by_llm', kb_id, doc_ids)

    def register_schema_set(self, schema_set: Type[BaseModel], kb_id: Optional[str] = DEFAULT_KB_ID,
                            force_refresh: bool = False) -> str:
        """
手动注册一个 Pydantic Model 作为当前算法的字段集合（schema），并绑定到指定知识库。
如果该知识库已绑定其他 schema，默认会报错；传入 ``force_refresh=True`` 则会替换旧绑定并清理旧数据。

Args:
    schema_set (Type[BaseModel]): 要注册的 Pydantic 模型，用作 schema 定义。
    kb_id (Optional[str]): 目标知识库 ID，默认为 ``DEFAULT_KB_ID``。
    force_refresh (bool): 若已有绑定，是否强制刷新并覆盖。默认 ``False``。

Returns:
    str: 生成的 schema_set_id。
"""
        return self._forward('_register_schema_set', schema_set, kb_id, force_refresh)

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, numbers: Optional[Set] = None
                  ) -> List[DocNode]:
        """按条件获取节点列表。

Args:
    uids (Optional[List[str]]): 指定节点 uid 列表。
    doc_ids (Optional[Set]): 指定文档 id 集合。
    group (Optional[str]): 节点组名。
    kb_id (Optional[str]): 知识库 id。
    numbers (Optional[Set]): 节点编号集合。

**Returns:**

- List[DocNode]: 命中的节点列表。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import Document
    >>> doc = Document()
    >>> nodes = doc.get_nodes(doc_ids={'doc_1'}, group='CoarseChunk', kb_id='kb_1', numbers={1, 2})
    """
        return self._forward('_get_nodes', uids, doc_ids, group, kb_id, numbers)

    def get_window_nodes(self, node: DocNode, span: tuple[int, int] = (-5, 5),
                         merge: bool = False) -> Union[List[DocNode], DocNode]:
        """获取指定节点在同一文档内的窗口节点。

Args:
    node (DocNode): 目标节点。
    span (tuple[int, int]): 窗口范围，基于 node.number 的相对偏移。
    merge (bool): 是否将窗口节点合并为一个节点返回。

**Returns:**

- Union[List[DocNode], DocNode]: 窗口节点列表，或合并后的单节点。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import Document
    >>> doc = Document()
    >>> node = doc.get_nodes(doc_ids={'doc_1'}, group='CoarseChunk', kb_id='kb_1', numbers={10})[0]
    >>> window_nodes = doc.get_window_nodes(node, span=(-2, 2), merge=False)
    """
        return self._forward('_get_window_nodes', node, span, merge)

    def _get_post_process_tasks(self):
        return lazyllm.pipeline(lambda *a: self._forward('_lazy_init'))

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Document', manager=hasattr(self._manager, '_manager'),
                                 server=isinstance(self._manager._kbs, ServerModule))

class UrlDocument(ModuleBase):
    """UrlDocument类继承自ModuleBase，用于通过指定的URL和名称管理远程文档资源。
内部通过lazyllm的UrlModule代理实际调用，支持文档查找、检索和活跃节点分组查询。

Args:
    url (str): 远程文档资源的访问URL。
    name (str): 当前文档分组名称，用于标识文档分组。
"""
    def __init__(self, url: str, name: str = None):
        super().__init__()
        self._missing_keys = set(dir(Document)) - set(dir(UrlDocument))
        self._manager = lazyllm.UrlModule(url=ensure_call_endpoint(url))
        self._curr_group = name or DocListManager.DEFAULT_GROUP_NAME

    def _forward(self, func_name: str, *args, **kwargs):
        args = (self._curr_group, func_name, *args)
        return self._manager._call('__call__', *args, **kwargs)

    def find(self, target) -> Callable:
        """生成一个部分应用函数，用于在当前文档组中查找指定目标。

Args:
    target (str): 需要查找的目标标识。

**Returns:**

- Callable: 调用时会执行查找操作的部分应用函数。
"""
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw):
        return self._forward('retrieve', *args, **kw)

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, numbers: Optional[Set] = None
                  ) -> List[DocNode]:
        """按条件获取远程文档节点列表。

Args:
    uids (Optional[List[str]]): 指定节点 uid 列表。
    doc_ids (Optional[Set]): 指定文档 id 集合。
    group (Optional[str]): 节点组名。
    kb_id (Optional[str]): 知识库 id。
    numbers (Optional[Set]): 节点编号集合。

**Returns:**

- List[DocNode]: 命中的节点列表。
"""
        return self._forward('_get_nodes', uids, doc_ids, group, kb_id, numbers)

    def get_window_nodes(self, node: DocNode, span: tuple[int, int] = (-5, 5),
                         merge: bool = False) -> Union[List[DocNode], DocNode]:
        """获取远程文档中指定节点的窗口节点。

Args:
    node (DocNode): 目标节点。
    span (tuple[int, int]): 窗口范围，基于 node.number 的相对偏移。
    merge (bool): 是否将窗口节点合并为一个节点返回。

**Returns:**

- Union[List[DocNode], DocNode]: 窗口节点列表，或合并后的单节点。
"""
        return self._forward('_get_window_nodes', node, span, merge)

    @cached_property
    def active_node_groups(self):
        return self._forward('active_node_groups')

    def __getattr__(self, name):
        if name in self._missing_keys:
            raise RuntimeError(f'Document generated with url and name has no attribute `{name}`')

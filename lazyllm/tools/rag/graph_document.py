import weakref
from pathlib import Path
from lazyllm import ModuleBase
from lazyllm.tools.servers.graphrag.graphrag_server_module import GraphRagServerModule

from .document import Document


class GraphDocument(ModuleBase):
    """基于 GraphRAG 的知识图谱查询文档处理模块。

此类在 Document 实例基础之上提供了用于操作 GraphRAG（基于图的检索增强生成）的高级接口。它负责管理 GraphRAG 服务的生命周期，包括知识图谱的初始化、索引构建以及查询等能力。

Args:
    document (Document): 用于构建知识图谱的 Document 实例。GraphRAG 的知识图谱将被创建在 ``{document._manager._dataset_path}/.graphrag_kg`` 路径下。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import Document, GraphDocument, GraphRetriever
    >>> doc = Document(dataset_path='your_doc_path', name='test_graphrag')
    >>> graph_document = GraphDocument(doc)
    >>> graph_document.start()
    >>> user_input = input('Press Enter when files are ready in dataset path')
    >>> graph_document.init_graphrag_kg(regenerate_config=True)
    >>> # Now you need to edit $dataset_path/.graphrag_kg/settings.yaml
    >>> user_input = input('Press Enter when settings.yaml is ready')
    >>> graph_document.start_graphrag_index(override=True)
    >>> status_dict = graph_document.graphrag_index_status()
    >>> lazyllm.LOG.info(f'graphrag index status: {status_dict}')
    >>> # Wait until the index is completed
    >>> user_input = input('Press Enter to start graphrag retriever: ')
    >>> graph_retriever = GraphRetriever(graph_document)
    >>> your_query = input('Enter your query: ')
    >>> print(graph_retriever.forward(your_query))
    """
    def __init__(self, document: Document):
        super().__init__()
        self._kg_dir = str(Path(document._manager._dataset_path) / '.graphrag_kg')
        self._graphrag_server_module = GraphRagServerModule(kg_dir=self._kg_dir)
        self._graphrag_index_task_id = None
        self._document = document
        document._graph_document = weakref.ref(self)

    def start(self):
        self._graphrag_server_module.start()

    def stop(self):
        self._graphrag_server_module.stop()
        self._graphrag_index_task_id = None

    def init_graphrag_kg(self, regenerate_config: bool = True):
        """
初始化 GraphRAG 知识图谱目录并准备相关文件。该方法会将文档数据集中所有文件复制到 GraphRAG 的输入目录，并初始化 GraphRAG 的项目结构。文件会被追加 UUID 后缀以避免命名冲突。

Args:
    regenerate_config (bool, optional): 是否重新生成 GraphRAG 的配置文件。如果为 True，将覆盖已有的配置。默认值为 True。
"""
        m = self._graphrag_server_module
        kb_files = self._document._list_all_files_in_dataset()
        m.prepare_files(kb_files, regenerate_config=regenerate_config)

    def start_graphrag_index(self, override: bool = True) -> str:
        """
启动 GraphRAG 的索引构建过程。该方法会启动一个异步索引任务，根据已准备好的文件构建知识图谱。索引过程在后台运行，可通过 graphrag_index_status() 进行监控。

Args:
    override (bool, optional): 如果已有索引，是否覆盖重建。为 True 时会删除并重新创建现有索引。默认值为 True。
"""
        m = self._graphrag_server_module
        res = m.create_index(override=override)
        self._graphrag_index_task_id = res['task_id']
        return 'Success'

    def graphrag_index_status(self) -> dict:
        """
获取当前 GraphRAG 索引任务的状态。

**Returns:**

- dict: 包含索引任务状态信息的字典。
"""
        m = self._graphrag_server_module
        res = m.index_status(self._graphrag_index_task_id)
        return res

    def query(self, query: str) -> str:
        """
查询 GraphRAG 知识图谱。此方法会对已建立索引的知识图谱执行查询，并基于图的结构和关系返回答案。

Args:
    query (str): 用于搜索知识图谱的自然语言查询。

**Returns:**

- str: 查询问题的答案。
"""
        m = self._graphrag_server_module
        res = m.query(query)
        return res['answer']

    def __del__(self):
        self._graphrag_server_module.stop()
        self._document._graph_documents = None


class UrlGraphDocument(ModuleBase):
    """用于通过 URL 查询远程 GraphRAG 服务的轻量级封装。

此类提供了一个简化的接口，用于向已经部署并运行的 GraphRAG 服务进行查询。

Args:
    graphrag_url (str): 远程 GraphRAG 服务端点的基础 URL，应为 'http://hostname:port' 格式。
"""
    def __init__(self, graphrag_url: str):
        super().__init__()
        self._graphrag_server_url = graphrag_url

    def forward(self, *args, **kw):
        return GraphRagServerModule.query_by_url(self._graphrag_server_url, *args, **kw)

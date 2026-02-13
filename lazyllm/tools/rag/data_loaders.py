from typing import List, Optional, Dict, Union
from lazyllm import LOG
from lazyllm.common.common import once_wrapper

from .doc_node import DocNode, ImageDocNode
from .store import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .dataReader import SimpleDirectoryReader
from collections import defaultdict

type_mapping = {
    DocNode: LAZY_ROOT_NAME,
    ImageDocNode: LAZY_IMAGE_GROUP,
}

class DirectoryReader:
    """用于从文件目录加载和处理文档的目录读取器类。

此类提供从指定目录读取文档并将其转换为文档节点的功能。它支持本地和全局文件读取器，并且可以处理不同类型的文档，包括图像。

Args:
    input_files (Optional[List[str]]): 要读取的文件路径列表。如果为None，文件将在调用load_data方法时加载。
    local_readers (Optional[Dict]): 特定于此实例的本地文件读取器字典。键是文件模式，值是读取器函数。
    global_readers (Optional[Dict]): 在所有实例间共享的全局文件读取器字典。键是文件模式，值是读取器函数。


Examples:
    >>> from lazyllm.tools.rag.data_loaders import DirectoryReader
    >>> from lazyllm.tools.rag.readers import DocxReader, PDFReader
    >>> local_readers = {
    ...     "**/*.docx": DocxReader,
    ...     "**/*.pdf": PDFReader
    >>> }
    >>> reader = DirectoryReader(
    ...     input_files=["path/to/documents"],
    ...     local_readers=local_readers,
    ...     global_readers={}
    >>> )
    >>> documents = reader.load_data()
    >>> print(f"加载了 {len(documents)} 个文档")
    """
    def __init__(self, input_files: Optional[List[str]], local_readers: Optional[Dict] = None,
                 global_readers: Optional[Dict] = None) -> None:
        self._input_files = input_files
        self._local_readers, self._global_readers = local_readers, global_readers

    @once_wrapper
    def _lazy_init(self):
        self._reader = SimpleDirectoryReader(file_extractor={**self._global_readers, **self._local_readers})

    def load_data(self, input_files: Optional[List[str]] = None, metadatas: Optional[Dict] = None,
                  *, split_nodes_by_type: bool = False) -> List[DocNode]:
        """从指定的输入文件加载和处理文档。

此方法使用配置的文件读取器（本地和全局）从输入文件读取文档，将它们处理成文档节点，并可选地将图像节点与文本节点分离。

Args:
    input_files (Optional[List[str]]): 要读取的文件路径列表。如果为None，使用初始化时指定的文件。
    metadatas (Optional[Dict]): 与加载文档关联的额外元数据。
    split_nodes_by_type (bool): 是否将图像等其他节点与文本节点分离。如果为True，返回(text_nodes, image_nodes)的元组。如果为False，一起返回所有节点。

**Returns:**

- Union[List[DocNode], Tuple[List[DocNode], List[ImageDocNode]]]: 如果split_nodes_by_type为False，返回所有文档节点的列表。如果为True，返回包含文本节点和图像节点的元组。
"""
        self._lazy_init()
        input_files = input_files or self._input_files
        nodes: Union[List[DocNode], Dict[str, List[DocNode]]] = defaultdict(list) if split_nodes_by_type else []
        for doc in self._reader(input_files=input_files, metadatas=metadatas):
            doc._group = type_mapping.get(type(doc), LAZY_ROOT_NAME)
            nodes[doc._group].append(doc) if split_nodes_by_type else nodes.append(doc)
        if not nodes:
            raise ValueError(f'No nodes load from path {input_files}, please check your data path.')
        LOG.info('DirectoryReader loads data done!')
        return nodes

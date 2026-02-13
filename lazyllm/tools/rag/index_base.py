from .doc_node import DocNode
from abc import ABC, abstractmethod
from typing import List, Optional

class IndexBase(ABC):
    """用于实现索引系统的抽象基类，支持更新、删除和查询文档节点。
`class IndexBase(ABC)`
此抽象基类定义了索引系统的接口，要求子类实现更新、删除和查询文档节点的方法。


Examples:
    >>> from mymodule import IndexBase, DocNode
    >>> class MyIndex(IndexBase):
    ...     def __init__(self):
    ...         self.nodes = []
    ...     def update(self, nodes):
    ...         self.nodes.extend(nodes)
    ...         print(f"Updated nodes: {nodes}")
    ...     def remove(self, uids, group_name=None):
    ...         self.nodes = [node for node in self.nodes if node.uid not in uids]
    ...         print(f"Removed nodes with uids: {uids}")
    ...     def query(self, *args, **kwargs):
    ...         print("Querying nodes...")
    ...         return self.nodes
    >>> index = MyIndex()
    >>> doc1 = DocNode(uid="1", content="Document 1")
    >>> doc2 = DocNode(uid="2", content="Document 2")
    >>> index.update([doc1, doc2])
    Updated nodes: [DocNode(uid="1", content="Document 1"), DocNode(uid="2", content="Document 2")]
    >>> index.query()
    Querying nodes...
    [DocNode(uid="1", content="Document 1"), DocNode(uid="2", content="Document 2")]
    >>> index.remove(["1"])
    Removed nodes with uids: ['1']
    >>> index.query()
    Querying nodes...
    [DocNode(uid="2", content="Document 2")]
    """
    # TODO(chenjiahao): change params `nodes` to `segments`, index should be able to handle segments
    @abstractmethod
    def update(self, nodes: List[DocNode]) -> None:
        """更新索引内容。

该方法接收一组文档节点对象，并将其添加或更新到索引结构中。通常用于增量构建或刷新索引。

Args:
    nodes (List[DocNode]): 需要更新的文档节点列表。
"""
        pass

    @abstractmethod
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        """从索引中移除指定文档节点。

可根据唯一标识符列表删除索引中的文档节点，可选地指定组名称以限定范围。

Args:
    uids (List[str]): 需要移除的文档节点的唯一标识符列表。
    group_name (Optional[str]): 可选的组名称，用于限定要删除的范围。
"""
        pass

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        """执行索引查询。

根据传入的参数执行查询操作，返回匹配的文档节点列表。
**注意:** 当前方法为接口占位，子类需要实现具体逻辑。

Args:
    *args: 任意位置参数，用于查询条件。
    **kwargs: 任意关键字参数，用于查询条件。
"""
        pass

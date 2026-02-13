from .base import NodeTransform
from ..doc_node import RichDocNode, DocNode
from typing import List


class RichTransform(NodeTransform):
    """
将 `RichDocNode` 拆分为 `DocNode` 列表，并保留每个 `DocNode` 的元数据。
输入必须是 `RichDocNode` 实例。

Args:
    node (RichDocNode): 需要拆分的富文档节点。

Returns:
    List[DocNode]: 拆分后的节点列表。


Examples:

    >>> from lazyllm.tools.rag.transform import RichTransform
    >>> nodes = RichTransform().transform(rich_node)
    """
    __support_rich__ = True

    def _clone_node(self, n: DocNode) -> DocNode:
        new_node = DocNode(content=n.text, metadata=n.metadata,
                           global_metadata=n.global_metadata)
        new_node.excluded_embed_metadata_keys = n.excluded_embed_metadata_keys
        new_node.excluded_llm_metadata_keys = n.excluded_llm_metadata_keys
        return new_node

    def transform(self, node: RichDocNode, **kwargs) -> List[DocNode]:
        assert isinstance(node, RichDocNode), f'Expected RichDocNode, got {type(node)}'
        return [self._clone_node(sub_node) for sub_node in node.nodes]

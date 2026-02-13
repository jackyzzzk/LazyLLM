from typing import Dict, List, Optional, Union, Set

from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability


class HybridStore(LazyLLMStoreBase):
    """混合存储类，结合了分段存储和向量存储的功能。

Args:
    segment_store (LazyLLMStoreBase): 分段存储实例，用于存储文档的原始内容。
    vector_store (LazyLLMStoreBase): 向量存储实例，用于存储文档的向量表示。
"""
    capability = StoreCapability.ALL
    need_embedding = True
    supports_index_registration = False

    def __init__(self, segment_store: LazyLLMStoreBase, vector_store: LazyLLMStoreBase):
        self.segment_store: LazyLLMStoreBase = segment_store
        self.vector_store: LazyLLMStoreBase = vector_store

    @property
    def dir(self):
        return self.segment_store.dir

    @override
    def connect(self, *args, **kwargs):
        """连接到底层的分段存储和向量存储。

Args:
    *args: 传递给存储连接方法的位置参数。
    **kwargs: 传递给存储连接方法的关键字参数。
"""
        self.segment_store.connect(*args, **kwargs)
        self.vector_store.connect(*args, **kwargs)

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        """向存储中插入或更新数据。

Args:
    collection_name (str): 集合名称。
    data (List[dict]): 要插入或更新的数据列表，每个数据项都是一个字典。

**Returns:**

- bool: 操作成功返回True，否则返回False。
"""
        segments = [{k: v for k, v in segment.items() if k != 'embedding'} for segment in data]
        return self.segment_store.upsert(collection_name=collection_name, data=segments) and \
            self.vector_store.upsert(collection_name=collection_name, data=data)

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        """从存储中删除数据。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 删除条件，默认为None。
    **kwargs: 其他参数。

**Returns:**

- bool: 操作成功返回True，否则返回False。
"""
        return self.segment_store.delete(collection_name=collection_name, criteria=criteria, **kwargs) and \
            self.vector_store.delete(collection_name=collection_name, criteria=criteria, **kwargs)

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        """从存储中获取数据。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 查询条件，默认为None。
    **kwargs: 其他参数。

**Returns:**

- List[dict]: 返回符合条件的数据列表。

Raises:
    ValueError: 当向量存储中的uid在分段存储中找不到时抛出。
"""
        res_segments = self.segment_store.get(collection_name=collection_name, criteria=criteria, **kwargs)
        if not res_segments: return []
        uids = [item.get('uid') for item in res_segments]
        res_vectors = self.vector_store.get(collection_name=collection_name, criteria={'uid': uids}, **kwargs)

        data = {}
        for item in res_segments:
            data[item.get('uid')] = item
        for item in res_vectors:
            if item.get('uid') in data:
                data[item.get('uid')]['embedding'] = item.get('embedding')
            else:
                raise ValueError(f'[HybridStore - get] uid {item["uid"]} in vector store'
                                 ' but not found in segment store')
        return list(data.values())

    @override
    def search(self, collection_name: str, query: str, query_embedding: Optional[Union[dict, List[float]]] = None,
               topk: int = 10, filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        """在存储中搜索数据。

Args:
    collection_name (str): 集合名称。
    query (str): 搜索查询字符串。
    query_embedding (Optional[Union[dict, List[float]]]): 查询的向量表示，默认为None。
    topk (int): 返回的最大结果数量，默认为10。
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): 过滤条件，默认为None。
    embed_key (Optional[str]): 嵌入向量的键名，默认为None。
    **kwargs: 其他参数。

**Returns:**

- List[dict]: 返回搜索结果列表。
"""
        if embed_key:
            # vector store only give uid and score
            res = self.vector_store.search(collection_name=collection_name, query=query, query_embedding=query_embedding,
                                           topk=topk, filters=filters, embed_key=embed_key, **kwargs)
            if not res: return []
            uid2score = {item['uid']: item['score'] for item in res}
            uids = list(uid2score.keys())
            segments = self.segment_store.get(collection_name=collection_name, criteria={'uid': uids})
            uid2segment = {}
            for segment in segments:
                segment['score'] = uid2score.get(segment['uid'], 0)
                uid2segment[segment.get('uid')] = segment
            ordered = [uid2segment[uid] for uid in uids if uid in uid2segment]
            return ordered
        else:
            res = self.segment_store.search(collection_name=collection_name, query=query,
                                            topk=topk, filters=filters, **kwargs)
            return res

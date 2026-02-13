from typing import Optional, Any

class GlobalMetadataDesc:
    """用于描述全局元数据的说明符，包括其类型、可选的元素类型、默认值和大小限制。
`class GlobalMetadataDesc`
此类用于描述元数据的属性，例如类型、可选约束和默认值。支持标量和数组数据类型，并对某些类型指定特定的大小限制。

Args:
    data_type (int): 元数据的类型，以整数表示，代表不同的数据类型（例如 VARCHAR、ARRAY 等）。
    element_type (Optional[int]): 如果 `data_type` 是数组，则表示数组中每个元素的类型。默认为 `None`。
    default_value (Optional[Any]): 元数据的默认值。如果未提供，默认值为 `None`。
    max_size (Optional[int]): 元数据的最大大小或长度。如果 `data_type` 为 `VARCHAR` 或 `ARRAY`，则此属性为必填项。
"""
    # max_size MUST be set when data_type is DataType.VARCHAR or DataType.ARRAY
    def __init__(self, data_type: int, element_type: Optional[int] = None,
                 default_value: Optional[Any] = None, max_size: Optional[int] = None):
        self.data_type = data_type
        self.element_type = element_type
        self.default_value = default_value
        self.max_size = max_size

# ---------------------------------------------------------------------------- #
# RAG system metadata keys
RAG_KB_ID = 'kb_id'
RAG_DOC_ID = 'docid'
RAG_DOC_PATH = 'lazyllm_doc_path'
RAG_DOC_FILE_NAME = 'file_name'
RAG_DOC_FILE_TYPE = 'file_type'
RAG_DOC_FILE_SIZE = 'file_size'
RAG_DOC_CREATION_DATE = 'creation_date'
RAG_DOC_LAST_MODIFIED_DATE = 'last_modified_date'
RAG_DOC_LAST_ACCESSED_DATE = 'last_accessed_date'

RAG_SYSTEM_META_KEYS = set([RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID, RAG_DOC_FILE_NAME, RAG_DOC_FILE_TYPE,
                            RAG_DOC_FILE_SIZE, RAG_DOC_CREATION_DATE, RAG_DOC_LAST_MODIFIED_DATE,
                            RAG_DOC_LAST_ACCESSED_DATE])

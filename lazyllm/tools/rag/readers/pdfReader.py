import io
from pathlib import Path
from typing import List, Optional, Callable
from lazyllm.common import retry
from lazyllm.thirdparty import fsspec, pypdf
from lazyllm import LOG

from .readerBase import get_default_fs, is_default_fs
from ..doc_node import DocNode
from .readerBase import _RichReader

RETRY_TIMES = 3


class PDFReader(_RichReader):
    """用于读取 PDF 文件并提取其中的文本内容。

Args:
    split_doc (bool): 若为 True（默认），则解析为一个 `RichDocNode`，可以搭配 `RichTransform` 解析出带有页信息的节点；
        若为 False，则解析为一个纯文本的 `DocNode`。
    post_func (Optional[Callable[[List[DocNode]], List[DocNode]]]): 结果后处理函数，
        需返回 `List[DocNode]`，并会将 `extra_info` 写入每个节点的 `global_metadata`。
    return_trace (bool): 是否返回处理过程的 trace，默认为 True。
    return_full_document (bool, 已弃用): 此参数将在未来版本中删除，请使用 `split_doc` 替代。

Notes:
    当 `split_doc=True` 时返回 `RichDocNode`，否则返回 `DocNode`，两种情况都只返回一个节点。
    当 `split_doc=True` 时，强烈建议搭配 `RichTransform` 使用，可以解析出带有页信息等 metadata 的节点；
    如不使用 `RichTransform`，则解析出的节点会回退为纯文本节点。
"""
    def __init__(self, split_doc: bool = True,
                 post_func: Optional[Callable[[List[DocNode]], List[DocNode]]] = None,
                 return_trace: bool = True, *, return_full_document=None) -> None:
        if return_full_document is not None:
            LOG.warning('return_full_document is deprecated, please use split_doc instead')
            assert split_doc ^ return_full_document, \
                'split_doc and return_full_document cannot be both True or False'
            split_doc = not return_full_document
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)

    @retry(stop_after_attempt=3)
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        fs = fs or get_default_fs()
        with fs.open(file, 'rb') as fp:
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())
            pdf = pypdf.PdfReader(stream)
            num_pages = len(pdf.pages)
            docs = []
            if self._split_doc:
                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]
                    metadata = {'page_label': page_label}
                    docs.append(DocNode(text=page_text, metadata=metadata))
            else:
                text = '\n'.join(pdf.pages[page].extract_text() for page in range(num_pages))
                docs.append(DocNode(text=text))
            return docs

import importlib.util

from pathlib import Path
from typing import List, Optional
from lazyllm.thirdparty import fsspec

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode
from lazyllm.thirdparty import html2text, ebooklib
from lazyllm import LOG

class EpubReader(LazyLLMReaderBase):
    """用于读取 `.epub` 格式电子书的文件读取器。

继承自 `LazyLLMReaderBase`，只需实现 `_load_data` 方法，即可通过 `Document` 组件自动加载 `.epub` 文件中的内容。

注意：当前版本不支持通过 fsspec 文件系统（如远程路径）加载 epub 文件，若提供 `fs` 参数，将回退到本地文件读取。

**Returns:**

- List[DocNode]: 所有章节内容合并后的文本节点列表。
"""
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            LOG.warning('fs was specified but EpubReader doesn\'t support loading from '
                        'fsspec filesystems. Will load from local filesystem instead.')

        text_list = []

        spec = importlib.util.find_spec('ebooklib.epub')
        if spec is None:
            raise ImportError(
                'Please install ebooklib to use ebooklib module. '
                'You can install it with `pip install ebooklib`'
            )
        epub_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(epub_module)

        book = epub_module.read_epub(file, options={'ignore_ncs': True})

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_list.append(html2text.html2text(item.get_content().decode('utf-8')))
        text = '\n'.join(text_list)
        return [DocNode(text=text)]

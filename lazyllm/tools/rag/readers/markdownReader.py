import re
from pathlib import Path
from lazyllm.thirdparty import fsspec
from typing import List, Optional, Tuple

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class MarkdownReader(LazyLLMReaderBase):
    """用于读取和解析 Markdown 文件的模块。支持去除超链接和图片，按标题和内容将 Markdown 划分成若干文本段落节点。

Args:
    remove_hyperlinks (bool): 是否移除超链接，默认 True。
    remove_images (bool): 是否移除图片标记，默认 True。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
"""
    def __init__(self, remove_hyperlinks: bool = True, remove_images: bool = True, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def _markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split('\n')

        current_header = None
        current_lines = []
        in_code_block = False

        for line in lines:
            if line.startswith('```'): in_code_block = not in_code_block

            header_match = re.match(r'^#+\s', line)
            if not in_code_block and header_match:
                if current_header is not None or len(current_lines) > 0:
                    markdown_tups.append((current_header, '\n'.join(current_lines)))
                current_header = line
                current_lines.clear()
            else:
                current_lines.append(line)

        markdown_tups.append((current_header, '\n'.join(current_lines)))
        return [(key.strip() if key is not None else None, re.sub(r'<.*?>', '', value))
                for key, value in markdown_tups]

    def remove_images(self, content: str) -> str:
        """移除内容中形如 ![[...]] 的自定义图片标签。

Args:
    content (str): 输入的 markdown 内容。

**Returns:**

- str: 移除图片标签后的内容。
"""
        pattern = r'!{1}\[\[(.*)\]\]'
        return re.sub(pattern, '', content)

    def remove_hyperlinks(self, content: str) -> str:
        """移除 Markdown 超链接，将 [文本](链接) 转换为纯文本。

Args:
    content (str): 输入的 markdown 内容。

**Returns:**

- str: 移除超链接后的内容，仅保留链接文本。
"""
        pattern = r'\[(.*)\]\((.*)\)'
        return re.sub(pattern, r'\1', content)

    def _parse_tups(self, filepath: Path, errors: str = 'ignore',
                    fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[Tuple[Optional[str], str]]:
        fs = fs or fsspec.implementations.local.LocalFileSystem()

        with fs.open(filepath, encoding='utf-8') as f:
            content = f.read().decode(encoding='utf-8')

        if self._remove_hyperlinks: content = self.remove_hyperlinks(content)
        if self._remove_images: content = self.remove_images(content)
        return self._markdown_to_tups(content)

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        tups = self._parse_tups(file, fs=fs)
        results = [DocNode(
            content=[value if header is None else f'\n\n{header}\n{value}' for header, value in tups])]
        return results

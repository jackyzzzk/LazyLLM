import re
from pathlib import Path
from typing import Dict, List, Optional
from lazyllm.thirdparty import fsspec

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class IPYNBReader(LazyLLMReaderBase):
    """用于读取和解析 Jupyter Notebook (.ipynb) 文件的模块。将 notebook 转换成脚本文本后，按代码单元划分为多个文档节点，或合并为单一文本节点。

Args:
    parser_config (Optional[Dict]): 预留的解析器配置参数，当前未使用，默认为 None。
    concatenate (bool): 是否将所有代码单元合并成一个整体文本节点，默认为 False，即分割为多个节点。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
"""
    def __init__(self, parser_config: Optional[Dict] = None, concatenate: bool = False, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._parser_config = parser_config
        self._concatenate = concatenate

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if file.name.endswith('.ipynb'):
            try:
                import nbconvert
            except ImportError:
                raise ImportError('Please install nbconvert `pip install nbconvert`')

        if fs:
            with fs.open(file, encoding='utf-8') as f:
                doc_str = nbconvert.exporters.ScriptExporter().from_file(f)[0]
        else:
            doc_str = nbconvert.exporters.ScriptExporter().from_file(file)[0]

        splits = re.split(r'In\[\d+\]:', doc_str)
        splits.pop(0)

        if self._concatenate: docs = [DocNode(text='\n\n'.join(splits))]
        else: docs = [DocNode(text=s) for s in splits]

        return docs

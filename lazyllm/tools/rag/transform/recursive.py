from typing import List, Tuple
from functools import partial
from .character import CharacterSplitter
from .base import _UNSET

class RecursiveSplitter(CharacterSplitter):
    """
递归拆分文本。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量。
    keep_separator (bool): 是否保留分隔符在拆分后的文本中。默认为False。
    is_separator_regex (bool): 是否使用正则表达式作为分隔符。默认为False。
    separators (List[str]): 用于拆分的分隔符列表。默认为['

', '
', ' ', '']。如果你想按多个分隔符拆分，可以设置这个参数。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import RecursiveSplitter
    >>> splitter = RecursiveSplitter(separators=['

    ', '
    ', ' ', ''])
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="recursive", transform=RecursiveSplitter, chunk_size=1024, chunk_overlap=100)
    """
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 keep_separator: bool = _UNSET, is_separator_regex: bool = _UNSET,
                 separators: List[str] = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         keep_separator=keep_separator, is_separator_regex=is_separator_regex)
        separators = self._get_param_value('separators', separators, None)

        self._separators = separators if separators else ['\n\n', '\n', ' ', '']
        self._cached_recursive_split_fns = [
            partial(self._default_split, self._get_separator_pattern(sep))
            for sep in self._separators
        ] + [list]

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        character_split_fns = self._character_split_fns
        if character_split_fns == []:
            character_split_fns = self._cached_recursive_split_fns
        splits = []
        for split_fn in character_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False

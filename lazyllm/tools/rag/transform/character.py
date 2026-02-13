from functools import partial
import re
import inspect

from typing import List, Union, Tuple, Callable, Optional
from .base import _TextSplitterBase, _TokenTextSplitter, _Split, _UNSET

class CharacterSplitter(_TextSplitterBase):
    """
将文本按字符拆分。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int): 控制并行处理的线程/进程数量。
    separator (str): 用于拆分的分隔符。默认为' '。
    is_separator_regex (bool): 是否使用正则表达式作为分隔符。默认为False。
    keep_separator (bool): 是否保留分隔符在拆分后的文本中。默认为False。
    **kwargs: 传递给拆分器的额外参数。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, CharacterSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="characters", transform=CharacterSplitter, chunk_size=1024, chunk_overlap=100)
    """
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 separator: str = _UNSET, is_separator_regex: bool = _UNSET, keep_separator: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        separator = self._get_param_value('separator', separator, ' ')
        is_separator_regex = self._get_param_value('is_separator_regex', is_separator_regex, False)
        keep_separator = self._get_param_value('keep_separator', keep_separator, False)

        self._separator = separator
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator
        self._character_split_fns = []
        self._cached_sep_pattern = self._get_separator_pattern(self._separator)
        self._cached_default_split_fns = None

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits, is_sentence = self._get_splits_by_fns(text)

        if len(text_splits) == 1 and self._token_size(text_splits[0]) > chunk_size:
            token_splitter = _TokenTextSplitter(chunk_size=chunk_size, overlap=self._overlap)
            token_sub_texts = token_splitter.split_text(text_splits[0], metadata_size=0)
            return [
                _Split(s, is_sentence=is_sentence, token_size=self._token_size(s))
                for s in token_sub_texts
            ]

        results = []
        for segment in text_splits:
            token_size = self._token_size(segment)
            if token_size <= chunk_size:
                results.append(_Split(segment, is_sentence=is_sentence, token_size=token_size))
            else:
                sub_results = self._split(segment, chunk_size=chunk_size)
                results.extend(sub_results)

        return results

    def set_split_fns(self, split_fns: Union[Callable[[str], List[str]], List[Callable[[str], List[str]]]], bind_separator: bool = None):  # noqa: E501
        """
CharacterSplitter有默认的拆分函数，你也可以设置自己的拆分函数。
可以设置多个拆分函数，CharacterSplitter会按顺序使用这些函数，分隔符参数将失效。

Args:
    split_fns (List[Callable[[str], List[str]]]): 要使用的拆分函数列表。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import CharacterSplitter
    >>> splitter = CharacterSplitter(separator='
    ')
    >>> splitter.set_split_fns([lambda text: text.split(' '), lambda text: text.split('
    ')])
    >>> text = 'Hello, world!'
    >>> splits = splitter.split_text(text, metadata_size=0)
    >>> print(splits)
    """
        if not isinstance(split_fns, list):
            split_fns = [split_fns]
        self._character_split_fns = []
        for split_fn in split_fns:
            if bind_separator is None:
                sig = inspect.signature(split_fn)
                has_separator = 'separator' in sig.parameters
                should_bind = has_separator
            else:
                should_bind = bind_separator

            if should_bind:
                fn = partial(split_fn, separator=self._separator)
            else:
                fn = split_fn

            self._character_split_fns.append(fn)

    def add_split_fn(self, split_fn: Callable[[str], List[str]], index: Optional[int] = None, bind_separator: bool = None):  # noqa: E501
        """
添加一个拆分函数到CharacterSplitter。

Args:
    split_fn (Callable[[str], List[str]]): 要添加的拆分函数。
    index (Optional[int]): 要添加的拆分函数的位置。默认为最后一个位置。
    bind_separator (bool): 是否将分隔符绑定到拆分函数。默认为False。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import CharacterSplitter
    >>> splitter = CharacterSplitter(separator='
    ')
    >>> splitter.add_split_fn(lambda text: text.split(' '), index=0)
    >>> text = 'Hello, world!'
    >>> splits = splitter.split_text(text, metadata_size=0)
    >>> print(splits)
    """
        if bind_separator is None:
            sig = inspect.signature(split_fn)
            has_separator = 'separator' in sig.parameters
            should_bind = has_separator
        else:
            should_bind = bind_separator

        if should_bind:
            fn = partial(split_fn, separator=self._separator)
        else:
            fn = split_fn

        if index is None:
            self._character_split_fns.append(fn)
        else:
            self._character_split_fns.insert(index, fn)

    def clear_split_fns(self):
        """
清除CharacterSplitter的所有拆分函数，并使用默认的拆分函数。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import CharacterSplitter
    >>> splitter = CharacterSplitter(separator='
    ')
    >>> splitter.clear_split_fns()
    >>> text = 'Hello, world!'
    >>> splits = splitter.split_text(text, metadata_size=0)
    >>> print(splits)
    """
        self._character_split_fns = []

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        character_split_fns = self._character_split_fns
        if character_split_fns == []:
            if self._cached_default_split_fns is None:
                self._cached_default_split_fns = [
                    partial(self._default_split, self._cached_sep_pattern),
                    lambda t: t.split(' '),
                    list
                ]
            character_split_fns = self._cached_default_split_fns

        splits = []
        for split_fn in character_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False

    def _default_split(self, sep_pattern: Union[str, set[str]], text: str) -> List[str]:
        splits = re.split(sep_pattern, text)
        results = []
        if self._keep_separator:
            for i in range(0, len(splits) - 1, 2):
                if i + 1 < len(splits):
                    combined = splits[i] + splits[i + 1]
                    if combined:
                        results.append(combined)
            if len(splits) % 2 == 1 and splits[-1]:
                results.append(splits[-1])
        else:
            results = [split for split in splits if split]
        return results

    def _get_separator_pattern(self, separator: str) -> Union[str, set[str]]:
        lookaround_prefixes = ('(?=', '(?<!', '(?<=', '(?!')
        lookaround_pattern = re.compile(r'^\(\?(?:=|<=|!|<!)')

        is_lookaround = (
            self._is_separator_regex
            and (separator.startswith(lookaround_prefixes) or bool(lookaround_pattern.match(separator)))
        )

        if self._is_separator_regex or is_lookaround:
            sep_pattern = separator
        else:
            needs_escape = any(char in separator for char in r'\.^$*+?{}[]|()')
            sep_pattern = re.escape(separator) if needs_escape else separator

        if self._keep_separator:
            sep_pattern = f'({sep_pattern})'
        else:
            sep_pattern = f'(?:{sep_pattern})'

        return sep_pattern

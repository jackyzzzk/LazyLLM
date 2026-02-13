from pathlib import Path
from typing import Dict, List, Optional
from lazyllm.thirdparty import fsspec
import importlib
from lazyllm.thirdparty import pandas as pd
from lazyllm import LOG

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode


_FILL_FUNC_MAP = {
    'fillna': lambda df: df.fillna(''),
    'ffill': lambda df: df.ffill(),
    'bfill': lambda df: df.bfill(),
}

def _apply_fill(df: pd.DataFrame, fill_method: Optional[str]) -> pd.DataFrame:
    if fill_method not in _FILL_FUNC_MAP:
        LOG.warning(f'Unsupported fill method: {fill_method}, using fillna instead')
    return _FILL_FUNC_MAP.get(fill_method, _FILL_FUNC_MAP['fillna'])(df)


class PandasCSVReader(LazyLLMReaderBase):
    """用于读取 CSV 文件并使用 pandas 进行解析。

Args:
    concat_rows (bool): 是否将所有行拼接为一个文本块，默认为 True。
    col_joiner (str): 列之间的连接符。
    row_joiner (str): 行之间的连接符。
    pandas_config (Optional[Dict]): pandas.read_csv 的可选配置项。
    fill_method (Optional[str]): 缺失值填充策略，可选 'fillna'(默认) / 'ffill' / 'bfill'。
    return_trace (bool): 是否返回处理过程的 trace。
"""
    def __init__(self, concat_rows: bool = True, col_joiner: str = ', ', row_joiner: str = '\n',
                 pandas_config: Optional[Dict] = None, fill_method: Optional[str] = 'fillna',
                 return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config or {}
        self._fill_method = fill_method

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)

        df = _apply_fill(df, self._fill_method)
        text_list = df.apply(lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1).tolist()

        if self._concat_rows: return [DocNode(text=(self._row_joiner).join(text_list))]
        else: return [DocNode(text=text) for text in text_list]

class PandasExcelReader(LazyLLMReaderBase):
    """用于读取 Excel 文件（.xlsx），并将内容提取为文本。

Args:
    concat_rows (bool): 是否将所有行拼接为一个文本块。
    sheet_name (Optional[str]): 要读取的工作表名称。若为 None，则读取所有工作表。
    pandas_config (Optional[Dict]): pandas.read_excel 的可选配置项。
    fill_method (Optional[str]): 缺失值填充策略，可选 'fillna'(default) / 'ffill' / 'bfill'。
    return_trace (bool): 是否返回处理过程的 trace。
"""
    def __init__(self, concat_rows: bool = True, sheet_name: Optional[str] = None,
                 pandas_config: Optional[Dict] = None, fill_method: Optional[str] = 'fillna',
                 return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._concat_rows = concat_rows
        self._sheet_name = sheet_name
        self._pandas_config = pandas_config or {}
        self._fill_method = fill_method

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        openpyxl_spec = importlib.util.find_spec('openpyxl')
        if openpyxl_spec is not None: pass
        else: raise ImportError('Please install openpyxl to read Excel files. '
                                'You can install it with `pip install openpyxl`')

        if not isinstance(file, Path): file = Path(file)
        if fs:
            with fs.open(file) as f:
                dfs = pd.read_excel(f, self._sheet_name, **self._pandas_config)
        else:
            dfs = pd.read_excel(file, self._sheet_name, **self._pandas_config)

        def process_df(df: pd.DataFrame) -> List[DocNode]:
            df = _apply_fill(df, self._fill_method)
            text_list = (df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist())

            if self._concat_rows:
                return [DocNode(text='\n'.join(text_list))]
            return [DocNode(text=text) for text in text_list]

        dfs_list = [dfs] if isinstance(dfs, pd.DataFrame) else dfs.values()
        return [doc for df in dfs_list for doc in process_df(df)]

import os
import mimetypes
import multiprocessing
import fnmatch
import traceback
from tqdm import tqdm
from datetime import datetime
from functools import reduce
from itertools import repeat
from typing import Dict, Optional, List, Callable, Type, Union
from pathlib import Path, PurePosixPath, PurePath
from lazyllm.thirdparty import fsspec
from lazyllm import ModuleBase, LOG, config
from lazyllm.components.formatter.formatterbase import _lazyllm_get_file_list
from lazyllm.tools.rag.readers.readerBase import TxtReader, DefaultReader
from .doc_node import DocNode
from .readers import (ReaderBase, PDFReader, DocxReader, HWPReader, PPTXReader, ImageReader, IPYNBReader,
                      EpubReader, MarkdownReader, MboxReader, PandasCSVReader, PandasExcelReader, VideoAudioReader,
                      get_default_fs, is_default_fs)
from .transform import NodeTransform, FuncNodeTransform
from .global_metadata import (RAG_DOC_PATH, RAG_DOC_FILE_NAME, RAG_DOC_FILE_TYPE, RAG_DOC_FILE_SIZE,
                              RAG_DOC_CREATION_DATE, RAG_DOC_LAST_MODIFIED_DATE, RAG_DOC_LAST_ACCESSED_DATE)

def _file_timestamp_format(timestamp: float, include_time: bool = False) -> Optional[str]:
    try:
        if include_time:
            return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    except Exception:
        return None

class _DefaultFileMetadataFunc:
    def __init__(self, fs: Optional['fsspec.AbstractFileSystem'] = None):
        self._fs = fs or get_default_fs()

    def __call__(self, file_path: str) -> Dict:
        stat_result = self._fs.stat(file_path)

        try:
            file_name = os.path.basename(str(stat_result['name']))
        except Exception:
            file_name = os.path.basename(file_path)

        creation_date = _file_timestamp_format(stat_result.get('created'))
        last_modified_date = _file_timestamp_format(stat_result.get('mtime'))
        last_accessed_date = _file_timestamp_format(stat_result.get('atime'))
        default_meta = {
            RAG_DOC_PATH: file_path,
            RAG_DOC_FILE_NAME: file_name,
            RAG_DOC_FILE_TYPE: mimetypes.guess_type(file_path)[0],
            RAG_DOC_FILE_SIZE: stat_result.get('size'),
            RAG_DOC_CREATION_DATE: creation_date,
            RAG_DOC_LAST_MODIFIED_DATE: last_modified_date,
            RAG_DOC_LAST_ACCESSED_DATE: last_accessed_date,
        }

        return {meta_key: meta_value for meta_key, meta_value in default_meta.items() if meta_value is not None}

class SimpleDirectoryReader(ModuleBase):
    """
模块化的文档目录读取器，继承自 ModuleBase，支持从文件系统读取多种格式的文档并转换为标准化的 DocNode 。

该类支持直接指定文件列表或输入目录（二者互斥）。内置了对常见格式（如 PDF、DOCX、PPTX、图片、CSV、Excel、音视频等）的支持，也允许用户注册自定义的文件读取器。

Args:
    input_dir (Optional[str]): 输入目录路径。与 input_files 互斥。目录必须存在。
    input_files (Optional[List]): 直接指定的文件列表。与 input_dir 互斥。文件必须存在于指定路径或 `config['data_path']` 下。
    exclude (Optional[List]): 需要排除的文件模式列表。
    exclude_hidden (bool): 是否排除隐藏文件。默认为 True。
    recursive (bool): 是否递归读取子目录。默认为 False。
    encoding (str): 文本文件的编码格式。默认为 "utf-8"。
    filename_as_id (bool): 已弃用参数，不再使用。如果提供会打印警告日志。
    required_exts (Optional[List[str]]): 需要处理的文件扩展名白名单。仅处理这些扩展名的文件。
    file_extractor (Optional[Dict[str, Callable]]): 自定义文件读取器字典。键为文件名模式，值为读取器函数。
    fs (Optional[AbstractFileSystem]): 自定义文件系统。默认为系统的默认文件系统。
    metadata_genf (Optional[Callable[[str], Dict]]): 元数据生成函数，接收文件路径返回元数据字典。默认为内部实现 (_DefaultFileMetadataFunc)。
    num_files_limit (Optional[int]): 最大读取文件数量限制。超过时仅处理前 N 个文件。
    return_trace (bool): 是否返回处理过程追踪信息。默认为 False。
    metadatas (Optional[Dict]): 预定义的全局元数据字典，将附加到所有文档上。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools.dataReader import SimpleDirectoryReader
    >>> reader = SimpleDirectoryReader(input_dir="yourpath/",recursive=True,exclude=["*.tmp"],required_exts=[".pdf", ".docx"])
    >>> documents = reader.load_data()
    """
    default_file_readers: Dict[str, Type[ReaderBase]] = {
        '*.pdf': PDFReader,
        '*.docx': DocxReader,
        '*.hwp': HWPReader,
        '*.pptx': PPTXReader,
        '*.ppt': PPTXReader,
        '*.pptm': PPTXReader,
        '*.gif': ImageReader,
        '*.jpeg': ImageReader,
        '*.jpg': ImageReader,
        '*.png': ImageReader,
        '*.webp': ImageReader,
        '*.ipynb': IPYNBReader,
        '*.epub': EpubReader,
        '*.md': MarkdownReader,
        '*.mbox': MboxReader,
        '*.csv': PandasCSVReader,
        '*.xls': PandasExcelReader,
        '*.xlsx': PandasExcelReader,
        '*.mp3': VideoAudioReader,
        '*.mp4': VideoAudioReader,
        '*.txt': TxtReader,
        '*.xml': TxtReader,
    }

    def __init__(self, input_dir: Optional[str] = None, input_files: Optional[List] = None,
                 exclude: Optional[List] = None, exclude_hidden: bool = True, recursive: bool = False,
                 encoding: str = 'utf-8', filename_as_id: bool = False, required_exts: Optional[List[str]] = None,
                 file_extractor: Optional[Dict[str, Callable]] = None, fs: Optional['fsspec.AbstractFileSystem'] = None,
                 metadata_genf: Optional[Callable[[str], Dict]] = None, num_files_limit: Optional[int] = None,
                 return_trace: bool = False, metadatas: Optional[Dict] = None) -> None:
        super().__init__(return_trace=return_trace)

        self._fs = fs or get_default_fs()
        self._encoding = encoding
        self._exclude = exclude
        self._recursive = recursive
        self._exclude_hidden = exclude_hidden
        self._required_exts = required_exts
        self._num_files_limit = num_files_limit
        self._Path = Path if is_default_fs(self._fs) else PurePosixPath
        self._metadatas = metadatas
        self._input_files = self._get_input_files(input_dir, input_files)
        self._file_extractor = {**self.default_file_readers, **(file_extractor or {})}
        self._metadata_genf = metadata_genf or _DefaultFileMetadataFunc(self._fs)
        if filename_as_id: LOG.warning('Argument `filename_as_id` for DataReader is no longer used')

    def _get_input_files(self, input_dir, input_files):
        if input_files:
            assert not input_dir, 'Cannot provide files and dir at the same time'
            input_files = [os.path.join(config['data_path'], p) if not self._fs.isfile(p) else p for p in input_files]
            input_files = [self._Path(p) if p else (_ for _ in ()).throw(ValueError, f'File {p} does not exist.')
                           for p in input_files]
        elif input_dir:
            if not self._fs.isdir(input_dir):
                raise ValueError(f'Directory {input_dir} does not exist.')
            input_files = self._add_files(self._Path(input_dir))
        return input_files

    def _add_files(self, input_dir: Path) -> List[Path]:  # noqa: C901
        all_files = set()
        rejected_files = set()
        rejected_dirs = set()

        if self._exclude is not None:
            for excluded_pattern in self._exclude:
                if self._recursive:
                    excluded_glob = self._Path(input_dir) / self._Path('**') / excluded_pattern
                else:
                    excluded_glob = self._Path(input_dir) / excluded_pattern
                for file in self._fs.glob(str(excluded_glob)):
                    if self._fs.isdir(file):
                        rejected_dirs.add(self._Path(file))
                    else:
                        rejected_files.add(self._Path(file))

        file_refs: List[str] = []
        if self._recursive:
            file_refs = self._fs.glob(str(input_dir) + '/**/*')
        else:
            file_refs = self._fs.glob(str(input_dir) + '/*')

        for ref in file_refs:
            ref = self._Path(ref)
            is_dir = self._fs.isdir(ref)
            skip_hidden = self._exclude_hidden and self._is_hidden(ref)
            skip_bad_exts = (self._required_exts is not None and ref.suffix not in self._required_exts)
            skip_excluded = ref in rejected_files
            if not skip_excluded:
                if is_dir:
                    ref_parent_dir = ref
                else:
                    ref_parent_dir = self._fs._parent(ref)
                for rejected_dir in rejected_dirs:
                    if str(ref_parent_dir).startswith(str(rejected_dir)):
                        skip_excluded = True
                        LOG.warning(f'Skipping {ref} because it in parent dir '
                                    f'{ref_parent_dir} which is in {rejected_dir}.')
                        break

            if is_dir or skip_hidden or skip_bad_exts or skip_excluded:
                continue
            else:
                all_files.add(ref)

        new_input_files = sorted(all_files)

        if len(new_input_files) == 0:
            raise ValueError(f'No files found in {input_dir}.')
        if self._num_files_limit is not None and self._num_files_limit > 0:
            new_input_files = new_input_files[0: self._num_files_limit]

        LOG.debug(f'[SimpleDirectoryReader] Total files add: {len(new_input_files)}')

        LOG.info(f'input_files: {new_input_files}')
        return new_input_files

    def _is_hidden(self, path: Path) -> bool:
        return any(part.startswith('.') and part not in ['.', '..'] for part in path.parts)

    def _exclude_metadata(self, documents: List[DocNode]) -> List[DocNode]:
        for doc in documents:
            doc._excluded_embed_metadata_keys.extend(
                ['file_name', 'file_type', 'file_size', 'creation_date',
                 'last_modified_date', 'last_accessed_date', 'lazyllm_store_num'])
            doc._excluded_llm_metadata_keys.extend(
                ['file_name', 'file_type', 'file_size', 'creation_date',
                 'last_modified_date', 'last_accessed_date', 'lazyllm_store_num'])
        return documents

    @staticmethod
    def find_extractor_by_file(input_file: Path, file_extractor: Dict[str, Callable], pathm: PurePath = Path):
        """
根据文件名或后缀从文件读取器映射中选择合适的提取器（extractor）。

该函数首先尝试使用文件后缀进行直接匹配（如 `*.txt`），
若未命中，则会遍历 `file_extractor` 的模式键（如 `*.json`, `**/docs/*.md`），
使用 `fnmatch` 进行模糊匹配，找到最符合的读取器。
如果没有匹配项，将返回默认读取器 `DefaultReader`。

Args:
    input_file (Path): 输入文件路径。
    file_extractor (Dict[str, Callable]): 文件模式到提取器的映射表。
    pathm (PurePath): 路径处理模块，用于生成匹配模式，默认使用 `Path`。

**Returns:**

- Callable: 与文件匹配的提取器函数，若无匹配则返回 `DefaultReader`。
"""
        filename_lower = str(input_file).lower()
        file_suffix = filename_lower.split('.')[-1]
        if extractor := file_extractor.get(f'*.{file_suffix}'): return extractor

        for pattern, extractor in file_extractor.items():
            pt_lower = str(pathm(pattern)).lower()
            match_pattern = pt_lower if pt_lower.endswith('*') else os.path.join(str(pathm.cwd()).lower(), pt_lower)
            if pt_lower.startswith('*'):
                match_pattern = pt_lower
            else:
                base = str(pathm.cwd()).lower()
                match_pattern = os.path.join(base, pt_lower)
            if fnmatch.fnmatch(filename_lower, match_pattern):
                return extractor
        return DefaultReader

    @staticmethod
    def load_file(input_file: Path, metadata_genf: Callable[[str], Dict], file_extractor: Dict[str, Callable],
                  encoding: str = 'utf-8', pathm: PurePath = Path, fs: Optional['fsspec.AbstractFileSystem'] = None,
                  metadata: Optional[Dict] = None) -> List[DocNode]:
        """使用指定的 Reader 将单个文件加载为 `DocNode` 列表。

该方法会根据文件名模式匹配合适的读取器（reader），并遵循以下优先级生成元数据：
`用户提供 > reader 自动生成 > metadata_genf 生成`。
在配置允许的情况下支持回退到原始文本读取。

Args:
    input_file (Path): 要读取的文件路径。
    metadata_genf (Callable): 根据文件路径生成元数据的函数。
    file_extractor (Dict[str, Callable]): 文件扩展名模式与 reader 的映射表。
    encoding (str): 文件读取时使用的文本编码，默认为 "utf-8"。
    pathm (PurePath): 路径处理模块，支持本地或远程路径。
    fs (AbstractFileSystem): 可选文件系统对象，兼容 fsspec 抽象。
    metadata (Dict): 可选用户自定义元数据，优先于自动生成。

**Returns:**

- List[DocNode]: 从文件中提取的文档对象列表。
"""
        # metadata priority: user > reader > metadata_genf
        user_metadata: Dict = metadata or {}
        metadata_generated: Dict = metadata_genf(str(input_file)) if metadata_genf else {}
        rd = SimpleDirectoryReader.find_extractor_by_file(input_file, file_extractor, pathm)
        if isinstance(rd, type) and issubclass(rd, TxtReader):
            reader = rd(encoding=encoding)
        elif isinstance(rd, type):
            reader = rd()
        else:
            reader = rd
        kwargs = {'fs': fs} if fs and not is_default_fs(fs) else {}

        try:
            docs = reader(input_file, **kwargs)
        except Exception as e:
            LOG.error(f'Error loading file {input_file}, skip it!')
            LOG.error(f'message: {e}\n Traceback: {traceback.format_tb(e.__traceback__)}')
            return []
        docs = [docs] if isinstance(docs, DocNode) else [] if docs is None else docs

        for doc in docs:
            metadata = metadata_generated.copy()
            metadata.update(doc._global_metadata or {})
            metadata.update(user_metadata)
            doc._global_metadata = metadata

        if config['rag_filename_as_id']:
            for i, doc in enumerate(docs):
                doc._uid = f'{input_file!s}_index_{i}'
        return docs

    def _load_data(self, show_progress: bool = False, num_workers: Optional[int] = None,
                   fs: Optional['fsspec.AbstractFileSystem'] = None, metadatas: Optional[Dict] = None,
                   input_dir: Optional[str] = None, input_files: Optional[List] = None) -> List[DocNode]:
        documents, fs, metadatas = [], fs or self._fs, metadatas or self._metadatas
        process_file = self._get_input_files(input_dir, input_files) if input_dir or input_files else self._input_files

        if num_workers and num_workers >= 1:
            if num_workers > multiprocessing.cpu_count():
                LOG.warning('Specified num_workers exceed number of CPUs in the system. '
                            'Setting `num_workers` down to the maximum CPU count.')
            with multiprocessing.get_context('spawn').Pool(num_workers) as p:
                results = p.starmap(SimpleDirectoryReader.load_file,
                                    zip(process_file, repeat(self._metadata_genf), repeat(self._file_extractor),
                                        repeat(self._encoding), repeat(self._Path),
                                        repeat(self._fs), metadatas or repeat(None)))
                documents = reduce(lambda x, y: x + y, results)
        else:
            if show_progress:
                process_file = tqdm(self._input_files, desc='Loading files', unit='file')
            for input_file, metadata in zip(process_file, metadatas or repeat(None)):
                documents.extend(
                    SimpleDirectoryReader.load_file(
                        input_file=input_file, metadata_genf=self._metadata_genf, file_extractor=self._file_extractor,
                        encoding=self._encoding, pathm=self._Path, fs=self._fs, metadata=metadata))

        return self._exclude_metadata(documents)

    def forward(self, *args, **kwargs) -> List[DocNode]:
        return self._load_data(*args, **kwargs)

    @staticmethod
    def get_default_reader(file_ext: str) -> Callable[[Path, Dict], List[DocNode]]:
        """
根据文件扩展名获取默认的文件读取器（Reader）。

该函数通过文件扩展名（如 `.txt`、`.json`）在默认读取器映射表中查找对应的 Reader，
若未以 `"*."` 开头，会自动补全后缀格式（例如 `"txt"` → `"*.txt"`）。
常见的默认 Reader 包括纯文本读取器、JSON 读取器、Markdown 读取器等。

Args:
    file_ext (str): 文件扩展名或匹配模式（例如 `"txt"` 或 `"*.json"`）。

**Returns:**

- Callable[[Path, Dict], List[DocNode]]: 与该扩展名对应的读取器函数，若未匹配则返回 `None`。
"""
        if not file_ext.startswith('*.'): file_ext = '*.' + file_ext
        return SimpleDirectoryReader.default_file_readers.get(file_ext)

    @staticmethod
    def add_post_action_for_default_reader(file_ext: str, f: Callable[[DocNode], Union[DocNode, List[DocNode]]]):
        """
为默认 Reader 添加后处理函数（Post Action）。

该方法允许在默认文件读取器（Reader）完成文档解析后，对生成的 `DocNode`
进行自定义后处理（如文本清洗、节点拆分、结构调整等）。
若指定的扩展名没有默认读取器，会抛出 `KeyError` 异常。

后处理函数可以是以下类型之一：

1. 继承自 `NodeTransform` 的类；
2. 普通函数，接收一个 `DocNode` 并返回修改后的 `DocNode` 或列表；
3. 可实例化的类型，会自动创建实例。

Args:
    file_ext (str): 文件扩展名或匹配模式（例如 `"*.txt"`）。
    f (Callable[[DocNode], Union[DocNode, List[DocNode]]]): 后处理函数或节点转换类。

**Raises:**

- KeyError: 当指定文件扩展名没有默认 Reader 时抛出。
"""
        if not file_ext.startswith('*.'): file_ext = '*.' + file_ext
        if file_ext not in SimpleDirectoryReader.default_file_readers:
            raise KeyError(f'{file_ext} has no default reader, use Document.add_reader instead')

        reader = SimpleDirectoryReader.default_file_readers[file_ext]
        assert isinstance(reader, type) and issubclass(reader, ReaderBase)

        if isinstance(f, type): f = f()
        if not isinstance(f, NodeTransform):
            try: f('test')
            except Exception: pass
            else: f = FuncNodeTransform(f, trans_node=False)
        reader.post_action = staticmethod(f)


config.add('rag_filename_as_id', bool, False, 'RAG_FILENAME_AS_ID',
           description='Whether to use filename as id for RAG.')
config.add('use_fallback_reader', bool, True, 'USE_FALLBACK_READER',
           description='Whether to use fallback reader for RAG.')


class FileReader(object):
    """
文件内容读取器，主要功能是将多种格式的输入文件转换为拼接后的纯文本内容。

Args:
    input_files (Optional[List]):直接指定的文件列表。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools.dataReader import FileReader
    >>> reader = FileReader()
    >>> content = reader("yourpath/")
    """

    def __call__(self, input_files):
        file_list = _lazyllm_get_file_list(input_files)
        if isinstance(file_list, str) and file_list is not None:
            file_list = [file_list]
        if len(file_list) == 0:
            return []
        nodes = SimpleDirectoryReader(input_files=file_list)._load_data()
        txt = [node.get_text() for node in nodes]
        return '\n'.join(txt)

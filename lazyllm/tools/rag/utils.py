import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time

from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (Any, Callable, Dict, Generator, List, Optional, Set, Tuple,
                    Union)
from urllib.parse import urlsplit, urlunsplit
from numbers import Integral

import pydantic
import sqlalchemy
from lazyllm.thirdparty import fastapi
from filelock import FileLock
from pydantic import BaseModel
from sqlalchemy import Column, Row, bindparam, insert, select, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import DeclarativeBase, sessionmaker

import lazyllm
from lazyllm import config
from lazyllm.common import override
from lazyllm.common.queue import sqlite3_check_threadsafety
from lazyllm.thirdparty import tarfile

from .doc_node import DocNode, MetadataMode
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH
from .index_base import IndexBase
from pathlib import Path

# min(32, (os.cpu_count() or 1) + 4) is the default number of workers for ThreadPoolExecutor
config.add(
    'max_embedding_workers',
    int,
    min(32, (os.cpu_count() or 1) + 4),
    'MAX_EMBEDDING_WORKERS',
    description='The default number of workers for embedding in RAG.')

config.add('default_dlmanager', str, 'sqlite', 'DEFAULT_DOCLIST_MANAGER',
           description='The default document list manager for RAG.')

def gen_docid(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()


class KBDataBase(DeclarativeBase):
    pass


class KBOperationLogs(KBDataBase):
    __tablename__ = 'operation_logs'
    id = Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    log = Column(sqlalchemy.Text, nullable=False)
    created_at = Column(sqlalchemy.DateTime, default=sqlalchemy.func.now(), nullable=False)


DocPartRow = Row
class KBDocument(KBDataBase):
    __tablename__ = 'documents'

    doc_id = Column(sqlalchemy.Text, primary_key=True)
    filename = Column(sqlalchemy.Text, nullable=False, index=True)
    path = Column(sqlalchemy.Text, nullable=False)
    created_at = Column(sqlalchemy.DateTime, default=sqlalchemy.func.now(), nullable=False)
    last_updated = Column(sqlalchemy.DateTime, default=sqlalchemy.func.now(), onupdate=sqlalchemy.func.now())
    meta = Column(sqlalchemy.Text, nullable=True)
    status = Column(sqlalchemy.Text, nullable=False, index=True)
    count = Column(sqlalchemy.Integer, default=0)

class KBGroup(KBDataBase):
    __tablename__ = 'document_groups'

    group_id = Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    group_name = Column(sqlalchemy.String, nullable=False, unique=True)

DocMetaChangedRow = Row
GroupDocPartRow = Row

class KBGroupDocuments(KBDataBase):
    __tablename__ = 'kb_group_documents'

    id = Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    doc_id = Column(sqlalchemy.String, sqlalchemy.ForeignKey('documents.doc_id'), nullable=False)
    group_name = Column(sqlalchemy.String, sqlalchemy.ForeignKey('document_groups.group_name'), nullable=False)
    status = Column(sqlalchemy.Text, nullable=True)
    log = Column(sqlalchemy.Text, nullable=True)
    need_reparse = Column(sqlalchemy.Boolean, default=False, nullable=False)
    new_meta = Column(sqlalchemy.Text, nullable=True)
    # unique constraint
    __table_args__ = (sqlalchemy.UniqueConstraint('doc_id', 'group_name', name='uq_doc_to_group'),)


class DocPathParsingResult(BaseModel):
    doc_id: str
    success: bool
    msg: str
    is_new: bool = False

class DocListManager(ABC):
    """抽象基类，用于管理文档列表和监控文档目录变化。

Args:
    path:要监控的文档目录路径。
    name:管理器名称。
    enable_path_monitoring:启用路径监控。



Examples:

    >>> import lazyllm
    >>> from lazyllm.rag.utils import DocListManager
    >>> manager = DocListManager(path='your_file_path/', name="test_manager", enable_path_monitoring=False)
    >>> added_docs = manager.add_files([test_file_list])
    >>> manager.enable_path_monitoring(True)
    >>> deleted = manager.delete_files([delete_file_list])
    """
    DEFAULT_GROUP_NAME = '__default__'
    __pool__ = dict()

    class Status:
        all = 'all'
        waiting = 'waiting'
        working = 'working'
        success = 'success'
        failed = 'failed'
        deleting = 'deleting'
        # deleted is no longer used
        deleted = 'deleted'

    def __init__(self, path, name, enable_path_monitoring=True):
        self._path = path
        self._name = name
        lazyllm.LOG.info(f'DocManager use file-system monitoring worker: {enable_path_monitoring}')
        self._id = hashlib.sha256(f'{name}@+@{path}'.encode()).hexdigest()
        if not os.path.isabs(path):
            raise ValueError(f'path [{path}] is not an absolute path')

        self._init_sql()
        self._delete_nonexistent_docs_on_startup()

        self._monitor_thread = threading.Thread(target=self._monitor_directory_worker)
        self._monitor_thread.daemon = True
        self._monitor_continue = True
        self._enable_path_monitoring = enable_path_monitoring
        self._init_monitor_event = threading.Event()
        if self._enable_path_monitoring:
            self._monitor_thread.start()
            self._init_monitor_event.wait()

    def _delete_nonexistent_docs_on_startup(self):
        ids = [row[0] for row in self.list_kb_group_files(details=True)
               if not Path(row[1]).exists()]
        if ids: self.delete_files(ids)

    def __new__(cls, *args, **kw):
        if cls is not DocListManager:
            return super().__new__(cls)
        return super().__new__(__class__.__pool__[config['default_dlmanager']])

    def init_tables(self) -> 'DocListManager':
        """确保数据库表默认分组存在。
"""
        if not self.table_inited():
            self._init_tables()
        # in case of using after relase
        self.add_kb_group(DocListManager.DEFAULT_GROUP_NAME)
        return self

    def _monitor_directory(self) -> Set[str]:
        files_list = []
        for root, _, files in os.walk(os.path.abspath(self._path)):
            # Skip hidden directories
            path_parts = root.split(os.sep)
            if any(part.startswith('.') for part in path_parts if part):
                continue

            # Skip hidden files
            files = [file_path for file_path in files if not file_path.startswith('.')]
            files = [os.path.join(root, file_path) for file_path in files]
            files_list.extend(files)
        return set(files_list)

    # Actually it shoule be 'set_docs_status_deleting'
    def delete_files(self, file_ids: List[str]) -> List[DocPartRow]:
        """将与文件关联的知识库条目设为删除中，并由各知识库进行异步删除解析结果及关联记录。

Args:
    file_ids (list of str): 要删除的文件ID列表
"""
        document_list = self.update_file_status(file_ids, DocListManager.Status.deleting)
        self.update_kb_group(cond_file_ids=file_ids, new_status=DocListManager.Status.deleting)
        return document_list

    @abstractmethod
    def table_inited(self):
        """检查数据库中的 `documents` 表是否已初始化。此方法在访问数据库时确保线程安全。
判断数据库中是否存在 `documents` 表。

**Returns:**

- bool: 如果 `documents` 表存在，返回 `True`；否则返回 `False`。

说明:
    - 使用线程安全锁 (`self._db_lock`) 确保对数据库的安全访问。
    - 通过 `self._db_path` 连接 SQLite 数据库，并使用 `check_same_thread` 配置选项。
    - 执行 SQL 查询：`SELECT name FROM sqlite_master WHERE type='table' AND name='documents'` 来检查表是否存在。
"""
        pass

    @abstractmethod
    def _init_tables(self): pass

    @abstractmethod
    def validate_paths(self, paths: List[str]) -> Tuple[bool, str, List[bool]]:
        """验证一组文件路径，以确保它们可以被正常处理。
此方法检查提供的路径是否是新的、已处理的或当前正在处理的，并确保处理文档时不会发生冲突。

Args:
    paths (List[str]): 要验证的文件路径列表。

**Returns:**

- Tuple[bool, str, List[bool]]: 返回一个元组，包括：
    - `bool`: 如果所有路径有效，则返回 `True`；否则返回 `False`。
    - `str`: 表示成功或失败原因的消息。
    - `List[bool]`: 一个布尔值列表，每个元素对应一个路径是否为新路径（`True` 表示新路径，`False` 表示已存在）。

说明:
    - 如果任何文档仍在处理中或需要重新解析，该方法会返回 `False`，并附带相应的错误消息。
    - 方法通过数据库会话和线程安全锁 (`self._db_lock`) 检索文档状态信息。
    - 不安全状态包括 `working` 和 `waiting`。

"""
        pass

    @abstractmethod
    def update_need_reparsing(self, doc_id: str, need_reparse: bool):
        """更新 `KBGroupDocuments` 表中某个文档的 `need_reparse` 状态。
此方法设置指定文档的 `need_reparse` 标志，并可选限定到特定分组。

Args:
    doc_id (str): 要更新的文档ID。
    need_reparse (bool): `need_reparse` 标志的新值。

说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - 方法会立刻将更改提交到数据库。
"""
        pass

    @abstractmethod
    def list_files(self, limit: Optional[int] = None, details: bool = False,
                   status: Union[str, List[str]] = Status.all,
                   exclude_status: Optional[Union[str, List[str]]] = None):
        """从 `documents` 表中列出文件，并支持过滤、限制返回结果以及返回详细信息。
此方法根据指定的条件，从数据库中检索文件ID或详细文件信息。

Args:
    limit (Optional[int]): 返回的最大文件数量。如果为 `None`，则返回所有匹配的文件。
    details (bool): 是否返回详细的文件信息（`True`）或仅返回文件ID（`False`）。
    status (Union[str, List[str]]): 要包含的状态或状态列表，默认为所有状态。
    exclude_status (Optional[Union[str, List[str]]]): 要排除的状态或状态列表，默认为 `None`。

**Returns:**

- List: 如果 `details=False`，则返回文件ID列表；如果 `details=True`，则返回详细文件行的列表。

说明:
    - 该方法根据 `status` 和 `exclude_status` 条件动态构造查询。
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - 如果指定了 `limit`，查询会附加 `LIMIT` 子句。
"""
        pass

    @abstractmethod
    def get_docs(self, doc_ids: List[str]) -> List[KBDocument]:
        """从数据库中检索类型为 `KBDocument` 的文档对象，基于提供的文档 ID 列表。

Args:
    doc_ids (List[str]): 要获取的文档 ID 列表。

**Returns:**

- List[KBDocument]: 与提供的文档 ID 对应的 `KBDocument` 对象列表。如果没有找到文档，将返回空列表。

说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问的安全性。
    - 查询使用 SQL 的 `IN` 子句，通过 `doc_id` 字段进行过滤。
    - 如果 `doc_ids` 为空，函数将直接返回空列表，而不会查询数据库。
"""
        pass

    @abstractmethod
    def set_docs_new_meta(self, doc_meta: Dict[str, dict]):
        """批量更新文档的元数据。

Args:
    doc_meta (Dict[str, dict]): 文档ID到新元数据的映射字典。

"""
        pass

    @abstractmethod
    def fetch_docs_changed_meta(self, group: str) -> List[DocMetaChangedRow]:
        """获取指定组中元数据已更改的文档，并将其 `new_meta` 字段重置为 `None`。
此方法检索元数据已更改（即 `new_meta` 不为 `None`）的所有文档，基于提供的组名。检索后，会将这些文档的 `new_meta` 字段重置为 `None`。

Args:
    group (str): 用于过滤文档的组名。

**Returns:**

- List[DocMetaChangedRow]: 包含文档 `doc_id` 和 `new_meta` 字段的行列表，表示元数据已更改的文档。

说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - 方法通过 SQL `JOIN` 操作连接 `KBDocument` 和 `KBGroupDocuments` 表以检索相关行。
    - 在获取数据后，将受影响行的 `new_meta` 字段更新为 `None`，并将更改提交到数据库。
"""
        pass

    @abstractmethod
    def list_all_kb_group(self):
        """列出所有知识库分组的名称。

**Returns:**

- list: 知识库分组名称列表。
"""
        pass

    @abstractmethod
    def add_kb_group(self, name):
        """添加一个新的知识库分组。

Args:
    name (str): 要添加的分组名称。
"""
        pass

    @abstractmethod
    def list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False,
                            status: Union[str, List[str]] = Status.all,
                            exclude_status: Optional[Union[str, List[str]]] = None,
                            upload_status: Union[str, List[str]] = Status.all,
                            exclude_upload_status: Optional[Union[str, List[str]]] = None,
                            need_reparse: Optional[bool] = False):
        """列出指定知识库组中的文件。

Args:
    group (str): 用于过滤文件的 KB 组名。默认为 `None`。
    limit (Optional[int]): 返回的最大文件数量。如果为 `None`，则返回所有匹配的文件。
    details (bool): 返回详细的文件信息或仅返回文件 ID 和路径。
    status (Union[str, List[str]]): 包含在结果中的 KB 组状态或状态列表。默认为所有状态。
    exclude_status (Optional[Union[str, List[str]]): 从结果中排除的 KB 组状态或状态列表。默认为 `None`。
    upload_status (Union[str, List[str]]): 包含在结果中的文档上传状态或状态列表。默认为所有状态。
    exclude_upload_status (Optional[Union[str, List[str]]): 从结果中排除的文档上传状态或状态列表。默认为 `None`。
    need_reparse (Optional[bool]): 过滤需要重新解析的文件或不需要重新解析的文件。默认为 `None`。

**Returns:**

- List: 如果 `details=False`，返回包含 `(doc_id, path)` 的元组列表。
          如果 `details=True`，返回包含附加元数据的详细行列表。

说明:
    - 方法根据提供的过滤条件动态构建 SQL 查询。
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - 如果 `status` 或 `upload_status` 参数为列表，则会使用 SQL 的 `IN` 子句进行处理。
"""
        pass

    def add_files(
        self,
        files: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        status: Optional[str] = Status.waiting,
        batch_size: int = 64,
    ) -> List[DocPartRow]:
        """批量向文档列表中添加文件，可选附加元数据、状态，并支持分批处理。
此方法将文件列表添加到数据库中，并为每个文件设置可选的元数据和初始状态。文件会以批量方式处理以提高效率。在文件添加完成后，它们会自动关联到默认的知识库 (KB) 组。

Args:
    files (List[str]): 添加的文件路径列表。
    metadatas (Optional[List[Dict[str, Any]]]): 与文件对应的元数据字典列表。默认为 `None`。
    status (Optional[str]): 添加文件的初始状态。默认为 `Status.waiting`。
    batch_size (int): 每批处理的文件数量。默认为 64。

**Returns:**

- List[DocPartRow]: 包含已添加文件及其相关信息的 `DocPartRow` 对象列表。

说明:
    - 方法首先通过辅助函数 `_add_doc_records` 创建文档记录。
    - 文件添加后，会自动关联到默认的知识库组 (`DocListManager.DEFAULT_GROUP_NAME`)。
    - 批量处理确保在添加大量文件时具有良好的可扩展性。
"""
        documents = self._add_doc_records(files, metadatas, status, batch_size)
        if documents:
            self.add_files_to_kb_group([doc.doc_id for doc in documents], group=DocListManager.DEFAULT_GROUP_NAME)
        return documents

    @abstractmethod
    def _get_all_docs(self): pass

    @abstractmethod
    def _get_docs(self, to_be_added_doc_ids: List, to_be_deleted_doc_ids: List, filter_status_list: List): pass

    @abstractmethod
    def _add_doc_records(self, files: List[str], metadatas: Optional[List] = None,
                         status: Optional[str] = Status.waiting, batch_size: int = 64) -> List[DocPartRow]: pass

    @abstractmethod
    def delete_unreferenced_doc(self):
        """删除数据库中标记为 "删除中" 且不再被引用的文档。
此方法从数据库中删除满足以下条件的文档：
    1. 文档状态为 `DocListManager.Status.deleting`。
    2. 文档的引用计数 (`count`) 为 0。
"""
        pass

    @abstractmethod
    def get_docs_need_reparse(self, group: Optional[str] = None) -> List[KBDocument]:
        """获取需要重新解析 (`need_reparse=True`)的指定组中的文档。
此方法检索标记为需要重新解析 (`need_reparse=True`) 的文档，基于提供的组名。仅包含状态为 `success` 或 `failed` 的文档。

Args:
    group (str): 用于过滤文档的组名。

**Returns:**

- List[KBDocument]: 需要重新解析的 `KBDocument` 对象列表。

说明:
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - 查询通过 SQL `JOIN` 操作连接 `KBDocument` 和 `KBGroupDocuments` 表，并基于组名和重新解析状态进行过滤。
    - 仅状态为 `success` 或 `failed` 且 `need_reparse=True` 的文档会被检索出来。
"""
        pass

    @abstractmethod
    def get_existing_paths_by_pattern(self, file_path: str) -> List[str]:
        """根据给定的模式，检索符合条件的文档路径。
此方法从数据库中获取所有符合提供的 SQL `LIKE` 模式的文档路径。

Args:
    pattern (str): 用于过滤文档路径的 SQL `LIKE` 模式。例如，`%example%` 匹配包含单词 "example" 的路径。

**Returns:**

- List[str]: 符合给定模式的文档路径列表。如果没有匹配的路径，则返回空列表。

说明:
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - SQL 查询中的 `LIKE` 操作符用于对文档路径进行模式匹配。
"""
        pass

    @abstractmethod
    def update_file_message(self, fileid: str, **kw):
        """更新指定文件的消息。

Args:
    fileid (str): 文件ID。
    **kw: 需要更新的其他键值对。
"""
        pass

    @abstractmethod
    def update_file_status(self, file_ids: List[str], status: str,
                           cond_status_list: Union[None, List[str]] = None) -> List[DocPartRow]:
        """更新指定文件的状态。

Args:
    file_ids (list of str): 更新状态的文件ID列表。
    status (str): 目标状态。
    cond_status_list(Union[None, List[str]]):限制只更新处于这些状态的文档
"""
        pass

    @abstractmethod
    def add_files_to_kb_group(self, file_ids: List[str], group: str):
        """将文件添加到指定的知识库分组中。

Args:
    file_ids (list of str): 要添加的文件ID列表。
    group (str): 要添加的分组名称。
"""
        pass

    @abstractmethod
    def delete_files_from_kb_group(self, file_ids: List[str], group: str):
        """从指定的知识库分组中删除文件。

Args:
    file_ids (list of str): 要删除的文件ID列表。
    group (str): 分组名称。
"""
        pass

    @abstractmethod
    def get_file_status(self, fileid: str):
        """获取指定文件的状态。

Args:
    fileid (str): 文件ID。

**Returns:**

- tr: 文件的当前状态。
"""
        pass

    @abstractmethod
    def update_kb_group(self, cond_file_ids: List[str], cond_group: Optional[str] = None,
                        cond_status_list: Optional[List[str]] = None, new_status: Optional[str] = None,
                        new_need_reparse: Optional[bool] = None) -> List[GroupDocPartRow]:
        """更新指定知识库分组中的内容。

Args:
    cond_file_ids (list of str, optional): 过滤使用的文件ID列表，默认为None。
    cond_group (str, optional): 过滤使用的知识库分组名称，默认为None。
    cond_status_list (list of str, optional): 过滤使用的状态列表，默认为None。
    new_status (str, optional): 新状态, 默认为None。
    new_need_reparse (bool, optinoal): 新的是否需重解析标志。

**Returns:**

- list: 得到更新的列表list of (doc_id, group_name)
"""
        pass

    @abstractmethod
    def release(self):
        """释放当前管理器的资源。

"""
        pass

    @property
    def enable_path_monitoring(self):
        """启用或禁用文档管理器的路径监控功能。
此方法用于启用或禁用文档管理器的路径监控功能。当启用时，会启动一个监控线程处理与路径相关的操作；当禁用时，会停止该线程并等待它终止。

Args:
    val (bool): 启用或禁用路径监控。

说明:
    - 如果 `val` 为 `True`，路径监控功能会通过将 `_monitor_continue` 设置为 `True` 并启动 `_monitor_thread` 来启用。
    - 如果 `val` 为 `False`，路径监控功能会通过将 `_monitor_continue` 设置为 `False` 并等待 `_monitor_thread` 终止来禁用。
    - 方法在管理监控线程时确保线程操作是安全的。
"""
        return self._enable_path_monitoring

    @enable_path_monitoring.setter
    def enable_path_monitoring(self, val: bool):
        """启用或禁用文档管理器的路径监控功能。
此方法用于启用或禁用文档管理器的路径监控功能。当启用时，会启动一个监控线程处理与路径相关的操作；当禁用时，会停止该线程并等待它终止。

Args:
    val (bool): 启用或禁用路径监控。

说明:
    - 如果 `val` 为 `True`，路径监控功能会通过将 `_monitor_continue` 设置为 `True` 并启动 `_monitor_thread` 来启用。
    - 如果 `val` 为 `False`，路径监控功能会通过将 `_monitor_continue` 设置为 `False` 并等待 `_monitor_thread` 终止来禁用。
    - 方法在管理监控线程时确保线程操作是安全的。
"""
        self._enable_path_monitoring = (val is True)
        if val is True:
            self._monitor_continue = True
            self._monitor_thread.start()
        else:
            self._monitor_continue = False
            if self._monitor_thread.is_alive():
                self._monitor_thread.join()

    def _monitor_directory_worker(self):
        failed_files_count = defaultdict(int)
        docs_all = self._get_all_docs()

        previous_files = set([doc.path for doc in docs_all])
        skip_files = set()
        is_first_run = True
        while self._monitor_continue:
            # 1. Scan files in the directory, find added and deleted files
            current_files = set(self._monitor_directory())
            to_be_added_files = current_files - previous_files - skip_files
            to_be_deleted_files = previous_files - current_files - skip_files
            failed_files = set()

            to_be_added_doc_ids = set([gen_docid(ele) for ele in to_be_added_files])
            to_be_deleted_doc_ids = set([gen_docid(ele) for ele in to_be_deleted_files])
            failed_doc_ids = set()
            filter_status_list = [DocListManager.Status.success,
                                  DocListManager.Status.failed, DocListManager.Status.waiting]

            docs_not_expected, docs_expected = self._get_docs(to_be_added_doc_ids,
                                                              to_be_deleted_doc_ids, filter_status_list)

            # 2. Skip new files that are already in the database
            failed_files.update([doc.path for doc in docs_not_expected])
            failed_doc_ids.update([doc.doc_id for doc in docs_not_expected])
            to_be_added_files -= failed_files
            # Actually it is add to doc with success status, then add to kb_group with waiting status
            self.add_files(list(to_be_added_files), status=DocListManager.Status.success)

            # 3. Skip deleted files that are: 1. not in the database, 2. status not success/failed/waiting
            safe_to_delete_files = set([doc.path for doc in docs_expected])
            safe_to_delete_doc_ids = set([doc.doc_id for doc in docs_expected])
            failed_doc_ids.update(to_be_deleted_doc_ids - safe_to_delete_doc_ids)
            failed_files.update(to_be_deleted_files - safe_to_delete_files)
            to_be_deleted_files = safe_to_delete_files
            to_be_deleted_doc_ids = safe_to_delete_doc_ids
            self.delete_files(list(to_be_deleted_doc_ids))

            # 4. update skip_files
            for ele in failed_files:
                failed_files_count[ele] += 1
                if failed_files_count[ele] >= 3:
                    skip_files.add(ele)
            # update previous files, while failed files will be re-processed in the next loop
            previous_files = (current_files | to_be_added_files) - to_be_deleted_files
            if is_first_run:
                self._init_monitor_event.set()
            is_first_run = False
            time.sleep(10)
        lazyllm.LOG.warning('END MONITORING')

    def __del__(self):
        self.enable_path_monitoring = False


class SqliteDocListManager(DocListManager):
    """基于 SQLite 的文档管理器，用于本地文件的持久化存储、状态管理与元信息追踪。

该类继承自 DocListManager，利用 SQLite 数据库存储文档记录。适用于管理具有唯一标识符的本地文档资源，并提供便捷的插入、查询、更新与状态过滤接口，支持可选的路径监控功能。

Args:
    path (str): 数据库存储路径。
    name (str): 数据库文件名（不包含路径）。
    enable_path_monitoring (bool): 是否启用对文件路径的变动监控，默认为 True。


Examples:
    >>> from lazyllm.tools.rag.utils import SqliteDocListManager
    >>> manager = SqliteDocListManager(path="./data", name="docs.sqlite")
    >>> manager.insert({"uid": "doc_001", "name": "example.txt", "status": "ready"})
    >>> print(manager.get("doc_001"))
    >>> files = manager.list_files(limit=5, details=True)
    >>> print(files)
    """
    def __init__(self, path, name, enable_path_monitoring=True):
        super().__init__(path, name, enable_path_monitoring)

    def _init_sql(self):
        root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
        os.makedirs(root_dir, exist_ok=True)
        self._db_path = os.path.join(root_dir, f'.lazyllm_dlmanager.{self._id}.db')
        self._db_lock = FileLock(self._db_path + '.lock')
        # ensure that this connection is not used in another thread when sqlite3 is not threadsafe
        self._check_same_thread = not sqlite3_check_threadsafety()
        self._engine = sqlalchemy.create_engine(
            f'sqlite:///{self._db_path}?check_same_thread={self._check_same_thread}'
        )
        self._Session = sessionmaker(bind=self._engine)
        self.init_tables()

    def _init_tables(self):
        KBDataBase.metadata.create_all(bind=self._engine)

    def table_inited(self):
        """检查数据库中是否已存在名为 "documents" 的表。

该方法通过查询 sqlite_master 元信息表，判断数据表是否已初始化。

**Returns:**

- bool: 如果 "documents" 表存在，返回 True；否则返回 False。
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute('SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'documents\'')
            return cursor.fetchone() is not None

    @staticmethod
    def get_status_cond_and_params(status: Union[str, List[str]],
                                   exclude_status: Optional[Union[str, List[str]]] = None,
                                   prefix: str = None):
        """生成用于文档状态筛选的 SQL 条件语句及其参数列表。

根据传入的包含状态和排除状态，构造 WHERE 子句中使用的 SQL 表达式。支持字段名前缀，用于联表查询等场景。

Args:
    status (str 或 list of str): 要包含的文档状态。若为 "all"，不添加包含条件。
    exclude_status (str 或 list of str, optional): 要排除的文档状态。不能为 "all"。
    prefix (str, optional): 字段名前缀（如联表查询中的别名），将应用于字段名。

**Returns:**

- Tuple[str, list]: 包含 SQL 条件语句和对应参数的元组。
"""
        conds, params = [], []
        prefix = f'{prefix}.' if prefix else ''
        if isinstance(status, str):
            if status != DocListManager.Status.all:
                conds.append(f'{prefix}status = ?')
                params.append(status)
        elif isinstance(status, (tuple, list)):
            conds.append(f'{prefix}status IN ({",".join("?" * len(status))})')
            params.extend(status)

        if isinstance(exclude_status, str):
            assert exclude_status != DocListManager.Status.all, 'Invalid status provided'
            conds.append(f'{prefix}status != ?')
            params.append(exclude_status)
        elif isinstance(exclude_status, (tuple, list)):
            conds.append(f'{prefix}status NOT IN ({",".join("?" * len(exclude_status))})')
            params.extend(exclude_status)

        return ' AND '.join(conds), params

    def _get_all_docs(self):
        with self._db_lock, self._Session() as session:
            return session.query(KBDocument).all()

    def _get_docs(self, to_be_added_doc_ids: List, to_be_deleted_doc_ids: List, filter_status_list: List):
        with self._db_lock, self._Session() as session:
            docs_not_expected = session.query(KBDocument).filter(KBDocument.doc_id.in_(to_be_added_doc_ids)).all()
            docs_expected = session.query(KBDocument).filter(KBDocument.doc_id.in_(to_be_deleted_doc_ids),
                                                             KBDocument.status.in_(filter_status_list)).all()
        return docs_not_expected, docs_expected

    def validate_paths(self, paths: List[str]) -> Tuple[bool, str, List[bool]]:
        """验证输入路径所对应的文档是否可以安全添加到数据库。

该方法会检查每个路径是否对应已有文档，若已存在，需判断其状态是否允许重解析。
若文档正在解析或等待解析，或上次重解析未完成，则视为不可用。

Args:
    paths (List[str]): 文件路径列表。

**Returns:**

- Tuple[bool, str, List[bool]]:
    - bool: 是否所有路径都验证通过。
    - str: 成功或失败的描述信息。
    - List[bool]: 与输入路径一一对应的布尔列表，表示该路径是否为新文档（True 为新文档，False 为已存在）。
        若验证失败，返回值为 None。
"""
        # check and return: success, msg, path_is_new for each path
        unsafe_staus_set = set([DocListManager.Status.working, DocListManager.Status.waiting])
        paths_is_new = [True] * len(paths)
        doc_ids = [gen_docid(path) for path in paths]
        doc_id_to_path = {doc_id: path for doc_id, path in zip(doc_ids, paths)}
        found_doc_ids = []
        found_doc_group_rows = []
        with self._db_lock, self._Session() as session:
            rows = session.execute(
                select(KBDocument.doc_id).where(KBDocument.doc_id.in_(doc_ids))
            ).fetchall()
            if len(rows) == 0:
                return True, 'Success', paths_is_new
            found_doc_ids = [row.doc_id for row in rows]
            found_doc_group_rows = session.execute(
                select(KBGroupDocuments.doc_id, KBGroupDocuments.need_reparse, KBGroupDocuments.status)
                .where(KBGroupDocuments.doc_id.in_(found_doc_ids))).fetchall()

        for doc_group_record in found_doc_group_rows:
            if doc_group_record.need_reparse:
                msg = f'Failed: {doc_id_to_path[doc_group_record.doc_id]} lasttime reparsing has not been finished'
                return False, msg, None
            if doc_group_record.status in unsafe_staus_set:
                return False, f'Failed: {doc_id_to_path[doc_group_record.doc_id]} is being parsed by kbgroup', None
        found_doc_ids = set(found_doc_ids)
        for i in range(len(paths)):
            cur_doc_id = doc_ids[i]
            if cur_doc_id in found_doc_ids:
                paths_is_new[i] = False
        return True, 'Success', paths_is_new

    def update_need_reparsing(self, doc_id: str, need_reparse: bool, group_name: Optional[str] = None):
        """更新指定文档的重解析标志位。

该方法用于设置某个文档是否需要重新解析。可以选择性地指定知识库分组进行精确匹配。

Args:
    doc_id (str): 文档的唯一标识符。
    need_reparse (bool): 是否需要重新解析文档。
    group_name (Optional[str]): 可选，所属的知识库分组名称。如果提供，将仅更新指定分组中的文档。
"""
        with self._db_lock, self._Session() as session:
            stmt = update(KBGroupDocuments).where(KBGroupDocuments.doc_id == doc_id)
            if group_name is not None: stmt = stmt.where(KBGroupDocuments.group_name == group_name)
            session.execute(stmt.values(need_reparse=need_reparse))
            session.commit()

    def list_files(self, limit: Optional[int] = None, details: bool = False,
                   status: Union[str, List[str]] = DocListManager.Status.all,
                   exclude_status: Optional[Union[str, List[str]]] = None):
        """列出文档数据库中符合状态条件的文件，并根据参数选择返回完整记录或仅返回文件路径。

Args:
    limit (Optional[int]): 要返回的记录数上限，若为 None 则返回所有符合条件的记录。
    details (bool): 是否返回完整的数据库行信息，若为 False 则仅返回文档路径（ID）。
    status (Union[str, List[str]]): 要包含在结果中的状态值，默认为包含所有状态。
    exclude_status (Optional[Union[str, List[str]]]): 要从结果中排除的状态值。

**Returns:**

- list: 文件记录列表或文档路径列表，具体取决于 `details` 参数。
"""
        query = 'SELECT * FROM documents'
        params = []
        status_cond, status_params = self.get_status_cond_and_params(status, exclude_status, prefix=None)
        if status_cond:
            query += f' WHERE {status_cond}'
            params.extend(status_params)
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall() if details else [row[0] for row in cursor]

    def get_docs(self, doc_ids: List[str]) -> List[KBDocument]:
        """根据给定的文档ID列表，从数据库中获取对应的文档对象列表。

Args:
    doc_ids (List[str]): 需要查询的文档ID列表。

**Returns:**

- List[KBDocument]: 匹配的文档对象列表。如果没有匹配项，返回空列表。
"""
        with self._db_lock, self._Session() as session:
            docs = session.query(KBDocument).filter(KBDocument.doc_id.in_(doc_ids)).all()
            return docs
        return []

    def set_docs_new_meta(self, doc_meta: Dict[str, dict]):
        """批量更新文档的元数据（meta），同时更新对应知识库分组中文档的 new_meta 字段（非等待状态的文档）。

Args:
    doc_meta (Dict[str, dict]): 字典，键为文档ID，值为对应的新元数据字典。
"""
        data_to_update = [{'_doc_id': k, '_meta': json.dumps(v)} for k, v in doc_meta.items()]
        with self._db_lock, self._Session() as session:
            # Use sqlalchemy core bulk update
            stmt = KBDocument.__table__.update().where(
                KBDocument.doc_id == bindparam('_doc_id')).values(meta=bindparam('_meta'))
            session.execute(stmt, data_to_update)
            session.commit()

            stmt = KBGroupDocuments.__table__.update().where(
                KBGroupDocuments.doc_id == bindparam('_doc_id'),
                KBGroupDocuments.status != DocListManager.Status.waiting).values(new_meta=bindparam('_meta'))
            session.execute(stmt, data_to_update)
            session.commit()

    def fetch_docs_changed_meta(self, group: str) -> List[DocMetaChangedRow]:
        """获取指定知识库分组中元数据发生变化的文档列表，并将对应的 new_meta 字段清空。

Args:
    group (str): 知识库分组名称。

**Returns:**

- List[DocMetaChangedRow]: 包含文档ID及其对应新元数据的列表。
"""
        rows = []
        conds = [KBGroupDocuments.group_name == group, KBGroupDocuments.new_meta.isnot(None)]
        with self._db_lock, self._Session() as session:
            rows = (
                session.query(KBDocument.doc_id, KBGroupDocuments.new_meta)
                .join(KBGroupDocuments, KBDocument.doc_id == KBGroupDocuments.doc_id)
                .filter(*conds).all()
            )
            stmt = update(KBGroupDocuments).where(sqlalchemy.and_(*conds)).values(new_meta=None)
            session.execute(stmt)
            session.commit()
        return rows

    def list_all_kb_group(self):
        """列出数据库中所有的知识库分组名称。

**Returns:**

- List[str]: 知识库分组名称列表。
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute('SELECT group_name FROM document_groups')
            return [row[0] for row in cursor]

    def add_kb_group(self, name):
        """向数据库中添加一个新的知识库分组名称，若已存在则忽略。

Args:
    name (str): 要添加的知识库分组名称。
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            conn.execute('INSERT OR IGNORE INTO document_groups (group_name) VALUES (?)', (name,))
            conn.commit()

    def list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False,
                            status: Union[str, List[str]] = DocListManager.Status.all,
                            exclude_status: Optional[Union[str, List[str]]] = None,
                            upload_status: Union[str, List[str]] = DocListManager.Status.all,
                            exclude_upload_status: Optional[Union[str, List[str]]] = None,
                            need_reparse: Optional[bool] = None):
        """列出指定知识库分组中的文件信息，可根据多种条件进行过滤。

Args:
    group (str, optional): 知识库分组名称，若为 None 则不按分组过滤。
    limit (int, optional): 限制返回的文件数量。
    details (bool): 是否返回详细的文件信息。
    status (str or List[str], optional): 过滤知识库分组中文件的状态。
    exclude_status (str or List[str], optional): 排除指定状态的文件。
    upload_status (str or List[str], optional): 过滤文件上传状态。
    exclude_upload_status (str or List[str], optional): 排除指定的上传状态。
    need_reparse (bool, optional): 是否只返回需要重新解析的文件。

**Returns:**

- list:
    - 如果 details 为 False，返回列表，每个元素为 (doc_id, path) 元组。
    - 如果 details 为 True，返回包含文件详细信息的元组列表，包括文档ID、路径、状态、元数据，
      知识库分组名、分组内状态及日志。
"""
        query = '''
            SELECT documents.doc_id, documents.path, documents.status, documents.meta,
                   kb_group_documents.group_name, kb_group_documents.status, kb_group_documents.log
            FROM kb_group_documents
            JOIN documents ON kb_group_documents.doc_id = documents.doc_id
        '''
        conds, params = [], []
        if group:
            conds.append('kb_group_documents.group_name = ?')
            params.append(group)

        if need_reparse is not None:
            conds.append('kb_group_documents.need_reparse = ?')
            params.append(int(need_reparse))

        status_cond, status_params = self.get_status_cond_and_params(status, exclude_status, prefix='kb_group_documents')
        if status_cond:
            conds.append(status_cond)
            params.extend(status_params)

        status_cond, status_params = self.get_status_cond_and_params(
            upload_status, exclude_upload_status, prefix='documents')
        if status_cond:
            conds.append(status_cond)
            params.extend(status_params)

        if conds: query += ' WHERE ' + ' AND '.join(conds)

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        if not details: return [row[:2] for row in rows]
        return rows

    def delete_unreferenced_doc(self):
        """删除数据库中标记为删除且未被任何知识库分组引用的文档记录。

该方法会查找状态为“deleting”且引用计数为0的文档，删除这些文档记录，并记录删除操作日志。

"""
        with self._db_lock, self._Session() as session:
            docs_to_delete = (
                session.query(KBDocument)
                .filter(KBDocument.status == DocListManager.Status.deleting, KBDocument.count == 0)
                .all()
            )
            for doc in docs_to_delete:
                session.delete(doc)
                log = KBOperationLogs(log=f'Delete obsolete file, doc_id:{doc.doc_id}, path:{doc.path}.')
                session.add(log)
            session.commit()

    def _add_doc_records(self, files: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                         status: Optional[str] = DocListManager.Status.waiting, batch_size: int = 64):
        documents = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else [None] * batch_size
            vals = []

            for i, file_path in enumerate(batch_files):
                doc_id = gen_docid(file_path)

                metadata = batch_metadatas[i].copy() if batch_metadatas[i] else {}
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, file_path)

                vals.append(
                    {
                        KBDocument.doc_id.name: doc_id,
                        KBDocument.filename.name: os.path.basename(file_path),
                        KBDocument.path.name: file_path,
                        KBDocument.meta.name: json.dumps(metadata),
                        KBDocument.status.name: status,
                        KBDocument.count.name: 0,
                    }
                )
            with self._db_lock, self._Session() as session:
                rows = session.execute(
                    insert(KBDocument)
                    .values(vals)
                    .prefix_with('OR IGNORE')
                    .returning(KBDocument.doc_id, KBDocument.path)
                ).fetchall()
                session.commit()
                documents.extend(rows)
        return documents

    def get_docs_need_reparse(self, group: str) -> List[KBDocument]:
        """获取指定知识库分组中需要重新解析的文档列表。

仅返回状态为“success”或“failed”的文档，且其对应的知识库分组记录标记为需要重新解析。

Args:
    group (str): 知识库分组名称。

**Returns:**

- List[KBDocument]: 需要重新解析的文档列表。
"""
        with self._db_lock, self._Session() as session:
            filter_status_list = [DocListManager.Status.success, DocListManager.Status.failed]
            documents = (
                session.query(KBDocument).join(KBGroupDocuments, KBDocument.doc_id == KBGroupDocuments.doc_id)
                .filter(KBGroupDocuments.need_reparse.is_(True),
                        KBGroupDocuments.group_name == group,
                        KBGroupDocuments.status.in_(filter_status_list)).all())
            return documents
        return []

    def get_existing_paths_by_pattern(self, pattern: str) -> List[str]:
        """根据路径匹配模式获取已存在的文档路径列表。

Args:
    pattern (str): 路径匹配模式，支持SQL的LIKE通配符。

**Returns:**

- List[str]: 匹配到的已存在文档路径列表。
"""
        exist_paths = []
        with self._db_lock, self._Session() as session:
            docs = session.query(KBDocument).filter(KBDocument.path.like(pattern)).all()
            exist_paths = [doc.path for doc in docs]
        return exist_paths

    # TODO(wangzhihong): set to metadatas and enable this function
    def update_file_message(self, fileid: str, **kw):
        """更新指定文件的字段信息。

Args:
    fileid (str): 文件的唯一标识符（doc_id）。
    **kw: 需要更新的字段及其对应的值，键值对形式传入。
"""
        set_clause = ', '.join([f'{k} = ?' for k in kw.keys()])
        params = list(kw.values()) + [fileid]
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            conn.execute(f'UPDATE documents SET {set_clause} WHERE doc_id = ?', params)
            conn.commit()

    def update_file_status(self, file_ids: List[str], status: str,
                           cond_status_list: Union[None, List[str]] = None) -> List[DocPartRow]:
        """更新多个文件的状态，支持根据当前状态进行条件过滤。

Args:
    file_ids (List[str]): 需要更新状态的文件ID列表。
    status (str): 要设置的新状态。
    cond_status_list (Union[None, List[str]], optional): 仅更新当前状态在此列表中的文件，默认为 None，表示不筛选。

**Returns:**

- List[DocPartRow]: 返回更新后的文件ID和路径列表。
"""
        rows = []
        if cond_status_list is None:
            sql_cond = KBDocument.doc_id.in_(file_ids)
        else:
            sql_cond = sqlalchemy.and_(KBDocument.status.in_(cond_status_list), KBDocument.doc_id.in_(file_ids))
        with self._db_lock, self._Session() as session:
            stmt = (
                update(KBDocument)
                .where(sql_cond)
                .values(status=status)
                .returning(KBDocument.doc_id, KBDocument.path)
            )
            rows = session.execute(stmt).fetchall()
            session.commit()
        return rows

    def add_files_to_kb_group(self, file_ids: List[str], group: str):
        """将多个文件添加到指定的知识库分组中。

该方法会将文件状态设置为等待处理（waiting），
若添加成功，则对应文档的计数（count）加一。

Args:
    file_ids (List[str]): 需要添加的文件ID列表。
    group (str): 知识库分组名称。
"""
        with self._db_lock, self._Session() as session:
            vals = []
            for doc_id in file_ids:
                vals = {
                    KBGroupDocuments.doc_id.name: doc_id,
                    KBGroupDocuments.group_name.name: group,
                    KBGroupDocuments.status.name: DocListManager.Status.waiting,
                }
                rows = session.execute(
                    insert(KBGroupDocuments).values(vals).prefix_with('OR IGNORE').returning(KBGroupDocuments.doc_id)
                ).fetchall()
                session.commit()
                if not rows:
                    continue
                doc = session.query(KBDocument).filter_by(doc_id=rows[0].doc_id).one()
                doc.count += 1
                session.commit()

    def delete_files_from_kb_group(self, file_ids: List[str], group: str):
        """从指定的知识库分组中删除多个文件。

删除成功后，对应文档的计数（count）减少，但不会低于0。
若文档不存在，会记录警告日志。

Args:
    file_ids (List[str]): 需要删除的文件ID列表。
    group (str): 知识库分组名称。
"""
        with self._db_lock, self._Session() as session:
            for doc_id in file_ids:
                records_to_delete = (
                    session.query(KBGroupDocuments)
                    .filter(KBGroupDocuments.doc_id == doc_id, KBGroupDocuments.group_name == group)
                    .all()
                )
                for record in records_to_delete:
                    session.delete(record)
                session.commit()
                if not records_to_delete:
                    continue
                try:
                    doc = session.query(KBDocument).filter_by(doc_id=records_to_delete[0].doc_id).one()
                    doc.count = max(0, doc.count - 1)
                    session.commit()
                except NoResultFound:
                    lazyllm.LOG.warning(f'No document found for {doc_id}')

    def get_file_status(self, fileid: str):
        """获取指定文件的状态。

Args:
    fileid (str): 文件的唯一标识符。

**Returns:**

- Optional[Tuple]: 返回包含状态的元组，若文件不存在则返回 None。
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute('SELECT status FROM documents WHERE doc_id = ?', (fileid,))
        return cursor.fetchone()

    def update_kb_group(self, cond_file_ids: List[str], cond_group: Optional[str] = None,
                        cond_status_list: Optional[List[str]] = None, new_status: Optional[str] = None,
                        new_need_reparse: Optional[bool] = None) -> List[GroupDocPartRow]:
        """更新知识库分组中指定文件的状态和重解析需求。

根据给定的文件ID列表、分组名及状态列表，批量更新对应文件在知识库分组中的状态及是否需要重解析标志。

Args:
    cond_file_ids (List[str]): 需要更新的文件ID列表。
    cond_group (Optional[str]): 分组名称，若指定则只更新该分组内的文件。
    cond_status_list (Optional[List[str]]): 仅更新状态匹配此列表的文件。
    new_status (Optional[str]): 新的文件状态。
    new_need_reparse (Optional[bool]): 新的重解析需求标志。

**Returns:**

- List[Tuple]: 返回更新后文件的doc_id、group_name及状态列表。
"""
        rows = []
        conds = []
        if not cond_file_ids:
            return rows
        conds.append(KBGroupDocuments.doc_id.in_(cond_file_ids))
        if cond_group is not None:
            conds.append(KBGroupDocuments.group_name == cond_group)
        if cond_status_list:
            conds.append(KBGroupDocuments.status.in_(cond_status_list))

        vals = {}
        if new_status is not None:
            vals[KBGroupDocuments.status.name] = new_status
        if new_need_reparse is not None:
            vals[KBGroupDocuments.need_reparse.name] = new_need_reparse

        if not vals:
            return rows
        with self._db_lock, self._Session() as session:
            stmt = (
                update(KBGroupDocuments)
                .where(sqlalchemy.and_(*conds))
                .values(vals)
                .returning(KBGroupDocuments.doc_id, KBGroupDocuments.group_name, KBGroupDocuments.status)
            )
            rows = session.execute(stmt).fetchall()
            session.commit()
        return rows

    def release(self):
        """清空数据库中的所有文档、分组及相关操作日志数据。

该操作会删除 documents、document_groups、kb_group_documents 和 operation_logs 表中的所有记录。
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            conn.execute('delete from documents')
            conn.execute('delete from document_groups')
            conn.execute('delete from kb_group_documents')
            conn.execute('delete from operation_logs')
            conn.commit()

    def __reduce__(self):
        return (__class__, (self._path, self._name, self._enable_path_monitoring))


DocListManager.__pool__ = dict(sqlite=SqliteDocListManager)


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description='API status code')
    msg: str = pydantic.Field('success', description='API status message')
    data: Any = pydantic.Field(None, description='API data')

    class Config:
        json_schema_extra = {
            'example': {
                'code': 200,
                'msg': 'success',
            }
        }


def run_in_thread_pool(func: Callable, params: Optional[List[Dict]] = None) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params or []:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):
            yield obj.result()


Default_Suport_File_Types = ['.docx', '.pdf', '.txt', '.json']


def _save_file(_file: 'fastapi.UploadFile', _file_path: str):
    file_content = _file.file.read()
    with open(_file_path, 'wb') as f:
        f.write(file_content)


def _convert_path_to_underscores(file_path: str) -> str:
    return file_path.replace('/', '_').replace('\\', '_')


def _save_file_to_cache(
    file: 'fastapi.UploadFile', cache_dir: str, suport_file_types: List[str]
) -> list:
    to_file_path = os.path.join(cache_dir, file.filename)

    sub_result_list_real_name = []
    if file.filename.endswith('.tar'):

        def unpack_archive(tar_file_path: str, extract_folder_path: str):

            out_file_names = []
            try:
                with tarfile.open(tar_file_path, 'r') as tar:
                    file_info_list = tar.getmembers()
                    for file_info in list(file_info_list):
                        file_extension = os.path.splitext(file_info.name)[-1]
                        if file_extension in suport_file_types:
                            tar.extract(file_info.name, path=extract_folder_path)
                            out_file_names.append(file_info.name)
            except tarfile.TarError as e:
                lazyllm.LOG.error(f'untar error: {e}')
                raise e

            return out_file_names

        _save_file(file, to_file_path)
        out_file_names = unpack_archive(to_file_path, cache_dir)
        sub_result_list_real_name.extend(out_file_names)
        os.remove(to_file_path)
    else:
        file_extension = os.path.splitext(file.filename)[-1]
        if file_extension in suport_file_types:
            if not os.path.exists(to_file_path):
                _save_file(file, to_file_path)
            sub_result_list_real_name.append(file.filename)
    return sub_result_list_real_name


def save_files_in_threads(
    files: List['fastapi.UploadFile'],
    override: bool,
    source_path,
    suport_file_types: List[str] = Default_Suport_File_Types,
):
    real_dir = source_path
    cache_dir = os.path.join(source_path, 'cache')

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    for dir in [real_dir, cache_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    param_list = [
        {'file': file, 'cache_dir': cache_dir, 'suport_file_types': suport_file_types}
        for file in files
    ]

    result_list = []
    for result in run_in_thread_pool(_save_file_to_cache, params=param_list):
        result_list.extend(result)

    already_exist_files = []
    new_add_files = []
    overwritten_files = []

    for file_name in result_list:
        real_file_path = os.path.join(real_dir, _convert_path_to_underscores(file_name))
        cache_file_path = os.path.join(cache_dir, file_name)

        if os.path.exists(real_file_path):
            if not override:
                already_exist_files.append(file_name)
            else:
                os.rename(cache_file_path, real_file_path)
                overwritten_files.append(file_name)
        else:
            os.rename(cache_file_path, real_file_path)
            new_add_files.append(file_name)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return (already_exist_files, new_add_files, overwritten_files)

# returns a list of modified nodes
def parallel_do_embedding(embed: Dict[str, Callable], embed_keys: Optional[Union[List[str], Set[str]]],  # noqa: C901
                          nodes: List[DocNode], group_embed_keys: Dict[str, List[str]] = None) -> List[DocNode]:
    if not nodes: return []

    max_workers = config['max_embedding_workers']

    tasks_by_key: Dict[str, List[DocNode]] = defaultdict(list)
    modified_nodes: List[DocNode] = []
    for node in nodes:
        if group_embed_keys and not group_embed_keys.get(node._group):
            continue
        keys_for_node = (group_embed_keys.get(node._group) if group_embed_keys else embed_keys) or embed.keys()
        miss = node.has_missing_embedding(keys_for_node)
        if not miss:
            continue
        modified_nodes.append(node)
        for k in miss:
            tasks_by_key[k].append(node)
            if hasattr(node, '_embedding_state'):
                node._embedding_state.add(k)

    if not tasks_by_key:
        return []
    concurrent_workers = min(max_workers, len(tasks_by_key))
    max_workers_per_key = max(1, max_workers // max(1, concurrent_workers))

    def _check_batch(fn):
        batch_size = getattr(fn, 'batch_size', None)
        return isinstance(batch_size, Integral) and batch_size > 1

    def _process_key(k: str, knodes: List[DocNode]):
        try:
            fn = embed[k]
            if _check_batch(fn):
                texts = [n.get_text(MetadataMode.EMBED) for n in knodes]
                vecs = fn(texts)
                if len(vecs) != len(texts):
                    raise ValueError(f'[LazyLLM - parallel_do_embedding][{k}] batch size mismatch: '
                                     f'[text_num:{len(texts)}] vs [vec_num:{len(vecs)}]')
                for n, v in zip(knodes, vecs):
                    n.set_embedding(k, v)
                return

            if max_workers_per_key == 1:
                for n in knodes:
                    n.do_embedding({k: fn})
            else:
                with ThreadPoolExecutor(max_workers=max_workers_per_key) as ex2:
                    futs = [ex2.submit(n.do_embedding, {k: fn}) for n in knodes]
                    for fut in as_completed(futs):
                        fut.result()
        except Exception as e:
            lazyllm.LOG.error(f'[LazyLLM - parallel_do_embedding][{k}] error: {e}')
            for n in knodes:
                if hasattr(n, '_embedding_state') and k in n._embedding_state:
                    with n._lock:
                        n._embedding_state.remove(k)
            raise e

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks_by_key))) as ex:
        futs = [ex.submit(_process_key, k, k_nodes) for k, k_nodes in tasks_by_key.items()]
        for fut in as_completed(futs):
            fut.result()
    return modified_nodes

class _FileNodeIndex(IndexBase):
    def __init__(self):
        self._file_node_map = {}  # Dict[path, Dict[uid, DocNode]]

    @override
    def update(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            path = node.global_metadata.get(RAG_DOC_PATH)
            if path:
                self._file_node_map.setdefault(path, {}).setdefault(node._uid, node)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        for path in list(self._file_node_map.keys()):
            uid2node = self._file_node_map[path]
            for uid in uids:
                uid2node.pop(uid, None)
            if not uid2node:
                del self._file_node_map[path]

    @override
    def query(self, files: List[str]) -> List[DocNode]:
        ret = []
        for file in files:
            nodes = self._file_node_map.get(file)
            if nodes:
                ret.extend(list(nodes.values()))
        return ret

def generic_process_filters(nodes: List[DocNode], filters: Dict[str, Union[str, int, List, Set]]) -> List[DocNode]:
    res = []
    for node in nodes:
        for name, candidates in filters.items():
            value = node.global_metadata.get(name)
            if (not isinstance(candidates, list)) and (not isinstance(candidates, set)):
                if value != candidates:
                    break
            elif (not value) or (value not in candidates):
                break
        else:
            res.append(node)
    return res

def sparse2normal(embedding: Union[Dict[int, float], List[Tuple[int, float]]], dim: int) -> List[float]:
    if not embedding:
        return []

    new_embedding = [0] * dim
    if isinstance(embedding, dict):
        for idx, val in embedding.items():
            new_embedding[int(idx)] = val
    elif isinstance(embedding, list) and isinstance(embedding[0], tuple):
        for pair in embedding:
            new_embedding[int(pair[0])] = pair[1]
    else:
        raise TypeError(f'unsupported embedding datatype `{type(embedding[0])}`')

    return new_embedding

def is_sparse(embedding: Union[Dict[int, float], List[Tuple[int, float]], List[float]]) -> bool:
    if isinstance(embedding, dict):
        return True

    if not isinstance(embedding, list):
        raise TypeError(f'unsupported embedding type `{type(embedding)}`')

    if len(embedding) == 0:
        raise ValueError('empty embedding type is not determined.')

    if isinstance(embedding[0], tuple):
        return True

    if isinstance(embedding[0], list):
        return False

    if isinstance(embedding[0], float) or isinstance(embedding[0], int):
        return False

    raise TypeError(f'unsupported embedding type `{type(embedding[0])}`')

def ensure_call_endpoint(raw: str, *, default_path: str = '/_call') -> str:
    if not raw:
        return raw

    raw = raw.strip()
    has_scheme = '://' in raw
    parts = urlsplit(raw if has_scheme else f'//{raw}', allow_fragments=True)

    if not parts.netloc:
        raise ValueError(f'Invalid endpoint (missing host): {raw}')

    scheme = parts.scheme or 'http'
    new_path = default_path
    return urlunsplit((scheme, parts.netloc, new_path, parts.query, parts.fragment))

def _get_default_db_config(db_name: str):
    db_name = db_name.split('.')[0]
    root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
    os.makedirs(root_dir, exist_ok=True)
    db_path = os.path.join(root_dir, f'lazyllm_{db_name}.db')
    return {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': db_path,
    }

def _orm_to_dict(obj) -> Dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

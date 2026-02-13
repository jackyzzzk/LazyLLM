import sqlite3
import threading
from abc import ABC, abstractmethod
from .globals import globals
from ..configs import config
import os
from typing import Type, Optional
from lazyllm.thirdparty import redis
from queue import Queue
from collections import deque
from filelock import FileLock

config.add('default_fsqueue', str, 'sqlite', 'DEFAULT_FSQUEUE',
           description='The default file system queue to use.')
config.add('fsqredis_url', str, '', 'FSQREDIS_URL',
           description='The URL of the Redis server for the file system queue.')
config.add('default_recent_k', int, 0, 'DEFAULT_RECENT_K',
           description='The number of recent inputs that RecentQueue keeps track of.')


class RecentQueue(Queue):
    """最近元素队列（RecentQueue）是对标准 Queue 的扩展实现，在不改变原有队列语义的前提下，额外维护一份「最近放入队列的 K 个元素」缓存。

该类常用于：
- 最近上下文记录（如 Prompt / 消息历史）
- 调试与问题定位时追踪最近输入
- 轻量级时间窗口缓存或回溯分析

参数说明：
- maxsize (int)：
    队列最大容量，语义与 queue.Queue 一致

- recent_k (Optional[int])：
    需要保留的最近元素数量：
    - None 或 0：不启用 recent 功能
    - 大于 0：最多保留 recent_k 个最近 put 进队列的元素


Examples:
    >>> from queue import Queue
    >>> from collections import deque
    >>> import threading

    >>> q = RecentQueue(maxsize=10, recent_k=3)
    >>> q.put("msg-1")
    >>> q.put("msg-2")
    >>> q.put("msg-3")

    >>> print(q.get_recent())
    ['msg-1', 'msg-2', 'msg-3']

    >>> q.put("msg-4")
    >>> print(q.get_recent())
    输出: ['msg-2', 'msg-3', 'msg-4']
    """
    def __init__(self, maxsize=0, recent_k=None):
        super().__init__(maxsize)
        self._recent_k = recent_k or config['default_recent_k']
        if self._recent_k:
            self._recent = deque(maxlen=self._recent_k)
            self._recent_lock = threading.Lock()

    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        if self._recent_k:
            with self._recent_lock:
                self._recent.append(item)
        return self

    def get_recent(self, join: Optional[str] = None, join_prefix: Optional[str] = None):
        """获取最近放入队列的元素。返回最近放入队列的元素列表或拼接后的字符串，不会消费（remove）队列中的任何元素。

重要说明：
- recent 是队列的旁路缓存（side-channel）
- 返回结果按时间顺序排列（从旧到新）

参数说明：
- join (Optional[str])：如果提供该参数，将 recent 列表中的元素使用 join 作为分隔符拼接为一个字符串。要求 recent 中的所有元素均为字符串类型。
- join_prefix (Optional[str])：当 join 生效且拼接结果非空时，在结果字符串前追加前缀，常用于上下文提示头。

返回值：
- List[Any]：当 join 为 None 时，返回 recent 元素的列表副本
- str：当 join 提供时，返回拼接后的字符串（可能带前缀）


Examples:
    >>> q = RecentQueue(recent_k=3)
    >>> q.put("a").put("b").put("c").put("d")
    >>> q.get_recent()
    ['b', 'c', 'd']

    >>> q.get_recent(join="\n", join_prefix="Recent:\n")
    'Recent:\nb\nc\nd'
    """
        r = []
        if self._recent_k:
            with self._recent_lock:
                r = list(self._recent)
        if join:
            assert isinstance(join, str), 'join symbol must be str'
            assert all([isinstance(s, str) for s in r]), 'all items of list to join must be str'
            r = join.join(r)
            if join_prefix and r: r = join_prefix + r
        return r


class FileSystemQueue(ABC):
    """基于文件系统的队列抽象基类。

FileSystemQueue是一个抽象基类，提供了基于文件系统的队列操作接口。它支持多种后端实现（如SQLite、Redis），用于在分布式环境中进行消息传递和数据流控制。

该类实现了单例模式，确保每个类名只有一个队列实例，并提供了线程安全的队列操作。

Args:
    klass (str, optional): 队列的类名标识符。默认为 ``'__default__'``。

**Returns:**

- FileSystemQueue: 队列实例（单例模式）
"""

    __queue_pool__ = dict()

    def __init__(self, *, klass='__default__'):
        super().__init__()
        self._class = klass

    def __new__(cls, *args, **kw):
        klass = kw.get('klass', '__default__')
        if klass not in __class__.__queue_pool__:
            if cls is __class__:
                __class__.__queue_pool__[klass] = cls.__default_queue__(*args, **kw)
            else:
                __class__.__queue_pool__[klass] = super().__new__(cls)
        return __class__.__queue_pool__[klass]

    @classmethod
    def get_instance(cls, klass):
        """获取指定类名对应的队列实例。

此方法会根据类名标识符返回对应的队列对象。如果该类名尚未注册，会触发自动初始化。

Args:
    klass (str): 队列类名标识符，不能为 ``'__default__'``。

**Returns:**

- FileSystemQueue: 与指定类名绑定的队列实例。
"""
        assert isinstance(klass, str) and klass != '__default__'
        return cls(klass=klass)

    @classmethod
    def set_default(cls, queue: Type):
        """设置默认的队列实现。

此方法用于指定默认的队列类，作为未传入 `klass` 参数时的后端实现。

Args:
    queue (Type): 默认队列类。
"""
        cls.__default_queue__ = queue

    @property
    def sid(self):
        return f'{globals._sid}-{self._class}'

    def enqueue(self, message):
        """将消息加入队列。

此方法将指定的消息添加到队列的尾部，遵循先进先出（FIFO）的原则。

Args:
    message: 要加入队列的消息内容。


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='enqueue_test')
    >>> queue.enqueue(123)
    >>> queue.peek()
    '123'
    """
        return self._enqueue(self.sid, message)

    def dequeue(self, limit=None):
        """从队列中取出消息。

此方法从队列头部取出消息并移除它们，可以指定一次取出的消息数量。

Args:
    limit (int, optional): 一次取出的最大消息数量。如果为None，则取出所有消息。默认为None。

**Returns:**

- list: 取出的消息列表。


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='dequeue_test')
    >>> for i in range(5):
    ...     queue.enqueue(f"Message{i}")
    >>> all_messages = queue.dequeue()
    >>> all_messages
    ['Message0', 'Message1', 'Message2', 'Message3', 'Message4']
    """
        return self._dequeue(self.sid, limit=limit)

    def peek(self):
        """获取队列中的下一个消息，但不移除。

**Returns:**

- Any: 队列中下一个可用的消息；如果队列为空，返回 ``None``。


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='peek_test')
    >>> queue.enqueue("First message")
    >>> queue.enqueue("Second message")
    >>> first_message = queue.peek()
    >>> first_message
    'First message'
    >>> queue.peek()
    'First message'
    """
        return self._peek(self.sid)

    def size(self):
        """获取队列中的消息数量。

**Returns:**

- int: 队列中当前消息的数量。


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='size_test')
    >>> queue.size()
    0
    >>> queue.enqueue("Message1")
    >>> queue.size()
    1
    >>> queue.enqueue("Message2")
    >>> queue.size()
    2
    >>> queue.dequeue()
    ['Message1', 'Message2']
    >>> queue.size()
    0
    """
        return self._size(self.sid)

    def init(self):
        """初始化队列。

该方法会清空当前队列中的所有消息，相当于调用 ``clear()``。
"""
        self.clear()

    def clear(self):
        """清空队列。

移除队列中的所有消息，使其恢复为空状态。


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='clear_test')
    >>> for i in range(10):
    ...     queue.enqueue(f"Message{i}")
    >>> queue.size()
    10
    >>> queue.clear()
    >>> queue.size()
    0
    >>> queue.peek() is None
    True
    """
        self._clear(self.sid)

    @abstractmethod
    def _enqueue(self, id, message): pass

    @abstractmethod
    def _dequeue(self, id, limit=None): pass

    @abstractmethod
    def _peek(self, id): pass

    @abstractmethod
    def _size(self, id): pass

    @abstractmethod
    def _clear(self, id): pass

# true means one connection can be used in multiple thread
# refer to: https://sqlite.org/compile.html#threadsafe
def sqlite3_check_threadsafety() -> bool:
    conn = sqlite3.connect(':memory:')
    res = conn.execute('''
        select * from pragma_compile_options
        where compile_options like 'THREADSAFE=%'
    ''').fetchall()
    conn.close()
    return True if res[0][0] == 'THREADSAFE=1' else False

class SQLiteQueue(FileSystemQueue):
    """基于 SQLite 的持久化文件系统队列。
该类扩展自 FileSystemQueue，使用 SQLite 数据库存储队列数据，通过 position 字段保证先进先出顺序，并支持并发安全的消息入队、出队、查看队头、队列大小查询和清空操作。
队列数据库默认存储在 ~/.lazyllm_filesystem_queue.db，通过文件锁机制确保多进程安全访问。

Args:
    klass (str): 队列分类名，用于逻辑隔离不同的队列，默认为 '__default__'。
"""
    def __init__(self, klass='__default__'):
        super(__class__, self).__init__(klass=klass)
        self.db_path = os.path.expanduser(os.path.join(config['home'], '.lazyllm_filesystem_queue.db'))
        self._lock = FileLock(self.db_path + '.lock')
        self._check_same_thread = not sqlite3_check_threadsafety()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock, sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS queue (
                id TEXT NOT NULL,
                position INTEGER NOT NULL,
                message TEXT NOT NULL,
                PRIMARY KEY (id, position)
            )
            ''')
            conn.commit()

    def _enqueue(self, id, message):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT MAX(position) FROM queue WHERE id = ?
                ''', (id,))
                max_pos = cursor.fetchone()[0]
                next_pos = 0 if max_pos is None else max_pos + 1
                cursor.execute('''
                INSERT INTO queue (id, position, message)
                VALUES (?, ?, ?)
                ''', (id, next_pos, message))
                conn.commit()

    def _dequeue(self, id, limit=None):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                if limit:
                    cursor.execute('SELECT message, position FROM queue WHERE id = ? '
                                   'ORDER BY position ASC LIMIT ?', (id, limit))
                else:
                    cursor.execute('SELECT message, position FROM queue WHERE id = ? '
                                   'ORDER BY position ASC', (id,))

                rows = cursor.fetchall()
                if not rows:
                    return []
                messages = [row[0] for row in rows]
                cursor.execute('DELETE FROM queue WHERE id = ? AND position IN '
                               f'({",".join([str(row[1]) for row in rows])})', (id, ))
                conn.commit()
                return messages

    def _peek(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT message FROM queue WHERE id = ? ORDER BY position ASC LIMIT 1
                ''', (id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return row[0]

    def _size(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT COUNT(*) FROM queue WHERE id = ?
                ''', (id,))
                return cursor.fetchone()[0]

    def _clear(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                DELETE FROM queue WHERE id = ?
                ''', (id,))
                conn.commit()


class RedisQueue(FileSystemQueue):
    """
基于 Redis 实现的文件系统队列（继承自 FileSystemQueue），用于跨进程/节点的消息传递与队列管理。内部使用指定的 redis_url 初始化并管理底层存储，同时提供线程安全的初始化逻辑。

Args:
    klass (str): 队列的分类名称，用于区分不同队列实例，默认值为 '__default__'。
"""
    def __init__(self, klass='__default__'):
        super(__class__, self).__init__(klass=klass)
        self.redis_url = config['fsqredis_url']
        self._lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            assert (
                conn.ping()
            ), 'Found fsque reids config but can not connect, please check your config `LAZYLLM_FSQREDIS_URL`.'
            if not conn.exists(self.sid):
                conn.rpush(self.sid, '<start>')

    def _enqueue(self, id, message):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            conn.rpush(id, message)

    def _dequeue(self, id, limit=None):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            if limit:
                limit = limit + 1
                vals = conn.lrange(id, 1, limit)
                conn.ltrim(id, limit, -1)
            else:
                vals = conn.lrange(id, 1, -1)
                conn.ltrim(id, 0, 0)
            if not vals:
                return []
            return [val.decode('utf-8') for val in vals]

    def _peek(self, id):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            val = conn.lindex(id, 1)
            if val is None:
                return None
            return val.decode('utf-8')

    def _size(self, id):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            rsize = conn.llen(id)
            return rsize - 1  # empty : [ <start> ]

    def _clear(self, id):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            conn.delete(id)

fsquemap = {
    'sqlite': SQLiteQueue,
    'redis': RedisQueue
}

FileSystemQueue.set_default(fsquemap.get(config['default_fsqueue'].lower()))

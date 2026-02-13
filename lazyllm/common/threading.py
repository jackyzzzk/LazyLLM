import threading
from queue import Queue
import functools
from .globals import globals
from concurrent.futures import ThreadPoolExecutor as TPE

def _sid_setter(sid):
    globals._init_sid(sid)

class Thread(threading.Thread):
    """LazyLLM 提供的增强线程类，继承自 Python 标准库的 `threading.Thread`。此类提供了额外的功能，包括会话ID管理、预钩子函数支持和异常处理机制。

Args:
    group: 线程组，默认为 ``None``
    target: 要在线程中执行的函数，默认为 ``None``
    name: 线程名称，默认为 ``None``
    args: 传递给目标函数的参数元组，默认为 ``()``
    kwargs: 传递给目标函数的关键字参数字典，默认为 ``None``
    prehook: 在线程执行前要调用的函数或函数列表，默认为 ``None``
    daemon: 是否为守护线程，默认为 ``None``


Examples:
    >>> import lazyllm
    >>> from lazyllm.common.threading import Thread
    >>> import time
    >>> def simple_task(name):
    ...     time.sleep(0.1)
    ...     return f"Hello from {name}"
    >>> thread = Thread(target=simple_task, args=("Worker",))
    >>> thread.start()
    >>> result = thread.get_result()
    >>> print(result)
    Hello from Worker
    >>> def setup_environment():
    ...     print("Setting up environment...")
    ...     return "environment_ready"
    >>> def validate_input(data):
    ...     print(f"Validating input: {data}")
    ...     if not isinstance(data, (int, float)):
    ...         raise ValueError("Input must be numeric")
    >>> def process_data(data):
    ...     print(f"Processing data: {data}")
    ...     time.sleep(0.1) 
    ...     return data * 2
    >>> thread = Thread(
    ...     target=process_data,
    ...     args=(42,),
    ...     prehook=[setup_environment, lambda: validate_input(42)]
    ... )
    >>> thread.start()
    Setting up environment...
    Validating input: 42
    Processing data: 42
    >>> result = thread.get_result()
    >>> print(f"Final result: {result}")
    Final result: 84
    """
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, prehook=None, daemon=None):
        self.q = Queue()
        if not isinstance(prehook, (tuple, list)): prehook = [prehook] if prehook else []
        prehook.insert(0, functools.partial(_sid_setter, sid=globals._sid))
        super().__init__(group, self.work, name, (prehook, target, args), kwargs, daemon=daemon)

    def work(self, prehook, target, args, **kw):
        """线程的核心工作方法，负责执行预钩子函数、目标函数，并处理异常和结果。

Args:
    prehook: 预钩子函数列表，在线程执行前调用
    target: 要执行的目标函数
    args: 传递给目标函数的参数
    **kw: 传递给目标函数的关键字参数

**注意**: 此方法由 `Thread` 类内部调用，用户通常不需要直接调用此方法。
"""
        [p() for p in prehook]
        try:
            r = target(*args, **kw)
        except Exception as e:
            self.q.put(e)
        else:
            self.q.put(r)

    def get_result(self):
        """获取线程执行结果的方法。此方法会阻塞直到线程执行完成，然后返回执行结果或重新抛出异常。

**Returns:**

- 线程执行的结果。如果目标函数正常执行，返回其返回值；如果发生异常，会重新抛出该异常。

**注意**: 此方法应该在调用 `thread.start()` 之后使用，用于获取线程的执行结果。
"""
        r = self.q.get()
        if isinstance(r, Exception):
            raise r
        return r


class ThreadPoolExecutor(TPE):
    def submit(self, fn, /, *args, **kwargs):
        def impl(sid, *a, **kw):
            globals._init_sid(sid)
            return fn(*a, **kw)

        return super(__class__, self).submit(functools.partial(impl, globals._sid), *args, **kwargs)

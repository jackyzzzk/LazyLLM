import multiprocessing
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor as PPE
import functools
import time
import atexit
from .utils import load_obj, dump_obj

@contextmanager
def _ctx(method='spawn'):
    m = multiprocessing.get_start_method()
    if m != method:
        multiprocessing.set_start_method(method, force=True)
    yield
    if m != method:
        multiprocessing.set_start_method(m, force=True)


class SpawnProcess(multiprocessing.Process):
    def start(self):
        """
使用spawn方式启动进程。

此方法在启动进程时强制使用spawn方式，这种方式会创建一个全新的Python解释器进程。spawn方式相比fork更安全，特别是在多线程环境下。

**说明:**
- 使用spawn方式启动新进程，避免了fork可能带来的问题
- 会临时切换启动方式为spawn，执行完后恢复原有启动方式
- 继承自multiprocessing.Process.start()的所有功能


Examples:

    ```python
    from lazyllm.common.multiprocessing import SpawnProcess

    def worker():
        print("Worker process running")

    # Create and start a process using spawn method
    process = SpawnProcess(target=worker)
    process.start()
    process.join()
    ```
    """
        with _ctx('spawn'):
            return super().start()


class ForkProcess(multiprocessing.Process):
    """LazyLLM 提供的增强进程类，继承自 Python 标准库的 `multiprocessing.Process`。此类专门使用 fork 启动方法来创建子进程，并提供了同步/异步执行模式的支持。

Args:
    group: 进程组，默认为 ``None``
    target: 要在进程中执行的函数，默认为 ``None``
    name: 进程名称，默认为 ``None``
    args: 传递给目标函数的参数元组，默认为 ``()``
    kwargs: 传递给目标函数的关键字参数字典，默认为 ``{}``
    daemon: 是否为守护进程，默认为 ``None``
    sync: 是否为同步模式，默认为 ``True``。在同步模式下，进程执行完目标函数后会自动退出；在异步模式下，进程会持续运行直到被手动终止。

**注意**: 此类主要用于 LazyLLM 内部的进程管理，特别是在需要长期运行的服务器进程中。


Examples:

    >>> import lazyllm
    >>> from lazyllm.common import ForkProcess
    >>> import time
    >>> import os
    >>> def simple_task(task_id):
    ...     print(f"Process {os.getpid()} executing task {task_id}")
    ...     time.sleep(0.1)  
    ...     return f"Task {task_id} completed by process {os.getpid()}"
    >>> process = ForkProcess(target=simple_task, args=(1,), sync=True)
    >>> process.start()
    Process 12345 executing task 1
    """
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, *, daemon=None, sync=True):
        super().__init__(group, ForkProcess.work(target, sync), name, args, kwargs or {}, daemon=daemon)

    @staticmethod
    def work(f, sync):
        """
ForkProcess 的核心工作方法，负责包装目标函数并处理同步/异步执行逻辑。

Args:
    f: 要执行的目标函数
    sync: 是否为同步模式。在同步模式下，执行完目标函数后进程会退出；在异步模式下，进程会持续运行。
"""
        def impl(*args, **kw):
            try:
                f(*args, **kw)
                if not sync:
                    while True: time.sleep(1)
            finally:
                atexit._run_exitfuncs()
        return impl

    def start(self):
        """
启动 ForkProcess 进程。此方法会使用 fork 启动方法来创建子进程，并开始执行目标函数。

此方法的特点：

- **Fork 启动**: 使用 fork 方法创建子进程，在 Unix/Linux 系统上提供更好的性能
- **上下文管理**: 自动管理进程启动方法的上下文，确保使用正确的启动方式
- **继承父类**: 继承自 `multiprocessing.Process.start()` 的所有功能

**注意**: 此方法会实际创建新的进程并开始执行，调用后进程会立即开始运行。

"""
        with _ctx('fork'):
            return super().start()


def _worker(f):
    return load_obj(f)()

class ProcessPoolExecutor(PPE):
    def submit(self, fn, /, *args, **kwargs):
        """
将任务提交到进程池中执行。

此方法将一个函数及其参数序列化后提交到进程池中执行，返回一个 `Future` 对象，用于获取任务执行结果或状态。

Args:
    fn (Callable): 要执行的函数。
    *args: 传递给函数的位置参数。
    **kwargs: 传递给函数的关键字参数。

**Returns:**

- concurrent.futures.Future: 表示任务执行状态的 `Future` 对象。


Examples:

    >>> from lazyllm.common.multiprocessing import ProcessPoolExecutor
    >>> import time
    >>> 
    >>> def task(x):
    ...     time.sleep(1)
    ...     return x * 2
    ... 
    >>> with ProcessPoolExecutor(max_workers=2) as executor:
    ...     future = executor.submit(task, 5)
    ...     result = future.result()
    ...     print(result)
    10
    """
        f = dump_obj(functools.partial(fn, *args, **kwargs))
        return super(__class__, self).submit(_worker, f)

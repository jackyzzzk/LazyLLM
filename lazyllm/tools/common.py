import lazyllm
from typing import Callable
import time

g_thread_pool = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config['thread_pool_worker_num'])

class StreamCallHelper:
    """流式调用辅助类，用于将阻塞调用包装为生成器形式，逐步返回执行结果。

Args:
    impl (Callable): 需要流式执行的函数或可调用对象。
    interval (float): 轮询队列的时间间隔，单位为秒，默认为0.1。
"""
    def __init__(self, impl: Callable, interval: float = 0.1):
        self._impl = impl
        self._sleep_interval = interval

    def __call__(self, *args, **kwargs):
        lazyllm.globals._init_sid()
        lazyllm.FileSystemQueue().clear()
        func_future = g_thread_pool.submit(self._impl, *args, **kwargs)
        need_continue = True
        str_total = ''
        while need_continue:
            if func_future.done():
                need_continue = False
            if value := lazyllm.FileSystemQueue().dequeue():
                str_streaming = ''.join(value)
                str_total += str_streaming
                yield str_streaming
            else:
                time.sleep(self._sleep_interval)
        result = func_future.result()
        if isinstance(result, str):
            if not str_total.endswith(result):
                yield result
        else:
            yield str(result)
        lazyllm.FileSystemQueue().clear()

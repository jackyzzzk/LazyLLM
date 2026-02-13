import os
import json
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED

from lazyllm import LOG, config
import lazyllm
from lazyllm.common.registry import LazyLLMRegisterMetaClass

config.add(
    'data_process_path', str, '', 'DATA_PROCESS_PATH',
    description='The path to store data files.')
config.add(
    'data_process_resume', bool, False, 'DATA_PROCESS_RESUME',
    description='Whether to resume data processing from saved state.')

class DataStateStore:
    def __init__(self, func_name, save_data=True):
        self.save_data = save_data
        self.resume = config['data_process_resume']
        self.save_path = None
        self.error_path = None
        self.progress_path = None
        self.buffer = []
        self.error_buffer = []
        self.indices_buffer = []
        self.processed_indices = set()
        self.is_done = False
        self.buffer_size_limit = 100
        self.last_save_time = time.time()
        self.lock = threading.RLock()

        if self.save_data:
            root = config['data_process_path'] or os.path.join(os.getcwd(), 'data_pipeline_res')
            save_folder = os.path.join(root, func_name)
            os.makedirs(save_folder, exist_ok=True)
            self.save_path = os.path.join(save_folder, f'{func_name}_results.jsonl')
            self.error_path = os.path.join(save_folder, f'{func_name}_error.jsonl')
            self.progress_path = f'{self.save_path}.json'
            self._init_files()

    def __getstate__(self):
        # Prevent pickling of the lock object
        state = self.__dict__.copy()
        if 'lock' in state:
            del state['lock']
        return state

    def __setstate__(self, state):
        # Restore state and re-initialize lock
        self.__dict__.update(state)
        self.lock = threading.RLock()

    def _init_files(self):
        with self.lock:
            if self.save_path and not self.resume:
                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                if os.path.exists(self.error_path):
                    os.remove(self.error_path)
                if os.path.exists(self.progress_path):
                    os.remove(self.progress_path)

    def load_progress(self):
        with self.lock:
            if self.save_path and self.resume and os.path.exists(self.progress_path):
                try:
                    with open(self.progress_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.is_done = data.get('is_done', False)
                        self.processed_indices = set(data.get('indices', []))
                except Exception:
                    LOG.warning(f'Failed to load progress from {self.progress_path}')
            else:
                self.processed_indices = set()
                self.is_done = False

    def save_results(self, results, indices=None, force=False):
        if not self.save_path:
            return

        with self.lock:
            if results:
                if isinstance(results, list):
                    self.buffer.extend(results)
                else:
                    self.buffer.append(results)

            if indices is not None:
                if indices == 'Done':
                    self.is_done = True
                    self.indices_buffer = []
                    force = True
                elif isinstance(indices, list):
                    self.indices_buffer.extend(indices)
                else:
                    self.indices_buffer.append(indices)

            current_time = time.time()
            time_diff = current_time - self.last_save_time

            # Intelligent storage: Adjust frequency based on buffer size and time
            if time_diff < 1.0:
                self.buffer_size_limit = min(self.buffer_size_limit * 2, 10000)
            elif time_diff > 10.0 and self.buffer_size_limit > 100:
                self.buffer_size_limit = max(self.buffer_size_limit // 2, 100)

            if force or (len(self.buffer) >= self.buffer_size_limit) or (time_diff > 5):
                self._flush()
                self.last_save_time = current_time

    def save_errors(self, errors):
        if not self.error_path:
            return
        with self.lock:
            if isinstance(errors, list):
                self.error_buffer.extend(errors)
            else:
                self.error_buffer.append(errors)

    def _flush(self):  # noqa: C901
        try:
            with self.lock:
                if self.buffer:
                    with open(self.save_path, 'a', encoding='utf-8') as f:
                        for res in self.buffer:
                            try:
                                line = json.dumps(res, ensure_ascii=False)
                                f.write(line + '\n')
                            except Exception as e:
                                LOG.warning(
                                    f'Could not serialize item to JSON in {self.save_path}. '
                                    f'Skipping. Item: {res}. Error: {e}')
                    self.buffer = []

                if self.error_buffer:
                    with open(self.error_path, 'a', encoding='utf-8') as f:
                        for res in self.error_buffer:
                            try:
                                line = json.dumps(res, ensure_ascii=False)
                                f.write(line + '\n')
                            except Exception as e:
                                LOG.warning(
                                    f'Could not serialize item to JSON in {self.error_path}. '
                                    f'Skipping. Item: {res}. Error: {e}')
                    self.error_buffer = []

                if self.indices_buffer:
                    self.processed_indices.update(self.indices_buffer)
                    self.indices_buffer = []

                if self.processed_indices or self.is_done:
                    with open(self.progress_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'is_done': self.is_done,
                            'indices': list(self.processed_indices)
                        }, f)
        except Exception as e:
            LOG.error(f'Failed to save results to {self.save_path}: {e}')

    def load_results(self):
        # Ensure any remaining buffer is flushed before loading
        self._flush()
        results = []
        with self.lock:
            if self.save_path and os.path.exists(self.save_path):
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            results.append(json.loads(line))
                        except Exception as e:
                            LOG.warning(f'Failed to parse line in {self.save_path}: {line.strip()}. Error: {e}')
        return results

class LazyLLMDataBase(metaclass=LazyLLMRegisterMetaClass):
    """数据处理算子基类。为注册到 data_register 的算子提供统一行为，包括并发执行、结果保存/恢复、进度记录和错误收集。

主要方法和行为：

- forward(self, input, **kwargs): 处理单条数据（子类/函数实现）。
- forward_batch_input(self, inputs, **kwargs): 处理批量数据并返回最终结果（子类/函数实现）。
- __call__(self, inputs): 统一入口，会根据子类是否实现 forward 或 forward_batch_input 选择执行逻辑；支持并发执行、断点续传和保存结果。
- set_output(self, path): 设置导出路径，调用后 __call__ 返回导出文件路径而不是内存结果。

构造函数参数:

- _concurrency_mode (str): 并发模式，'process'|'thread'|'single'。
- _save_data (bool): 是否保存中间结果到磁盘以便 Resume。
- _max_workers (int|None): 最大并发工作进程/线程数，None 表示使用默认。
- _ignore_errors (bool): 是否忽略任务异常。
- **kwargs (dict): 其它传递给算子的参数。

配置项（通过 lazyllm.config）:

- data_process_path (str): 存储处理结果的根路径。
- data_process_resume (bool): 是否开启 Resume 功能，从进度文件继续处理。


Examples:
    ```python
    from lazyllm.tools.data import LazyLLMDataBase

    # simple usage: subclass and implement forward
    class EchoOp(LazyLLMDataBase):
        def forward(self, data):
            return {'text': data.get('text', '')}

    op = EchoOp(_save_data=True)
    res = op([{'text': 'hello'}])  # returns list or exported path depending on set_output
    ```
    """
    def __init__(self, _concurrency_mode=None, _save_data=True, _max_workers=None,
                 _ignore_errors=True, **kwargs):
        self._concurrency_mode = _concurrency_mode or getattr(self, '_concurrency_mode', 'process')
        if _max_workers:
            self._max_workers = _max_workers
        elif self._concurrency_mode == 'process':
            self._max_workers = os.cpu_count()
        else:
            self._max_workers = min(max(32, (os.cpu_count() or 1) * 5), 128)
        self._ignore_errors = _ignore_errors
        self._store = DataStateStore(self.__class__.__name__, _save_data)
        self._lazyllm_kwargs = kwargs
        self._export_path = None

    def set_output(self, output_path):
        """设置输出路径，用于把最终结果导出为 jsonl 文件并返回文件路径。

Args:
    output_path (str): 文件夹路径或具体 .jsonl 文件路径。若为文件夹，则在该文件夹下创建以类名命名的 jsonl 文件。

行为：

- 如果传入的是文件夹路径，则在该文件夹下创建以类名命名的 jsonl 文件。
- 如果传入的是以 .jsonl 结尾的路径，则直接写入该文件（必要时会创建目录）。
- 返回写入的绝对路径字符串。


Examples:
    ```python
    from lazyllm.tools.data import Demo2

    # export to a directory (will create DemoClass.jsonl)
    op = Demo2.rich_content(input_key='text').set_output('./out_dir')
    path = op([{'text': 'sample'}])
    print(path)  # ./out_dir/RichContent.jsonl or similar

    # export to a specific file
    op = Demo2.rich_content(input_key='text').set_output('./out_dir/results.jsonl')
    path = op([{'text': 'sample'}])
    print(path)  # ./out_dir/results.jsonl
    ```
    """
        self._export_path = output_path
        return self

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f) or \
            getattr(self.__class__, '__reg_overwrite__', None) == f

    def forward(self, input_data, **kwargs):
        """子类需要实现的方法，处理单条数据。返回值支持：

- dict: 表示处理后的单条结果。
- list: 表示将一条输入展开为多条输出。
- None: 表示保留原始输入（不修改）。
- 抛出异常或返回错误对象会被记录到错误文件并跳过（依赖配置和调用者）。

Args:
    input (dict): 单条输入数据字典。
    **kwargs (dict): 其它用户传入的参数。


Examples:
    ```python
    from lazyllm.tools.data import LazyLLMDataBase

    class MyOp(LazyLLMDataBase):
        def forward(self, data):
            # return dict or list or None
            return {'text': data.get('text', '').upper()}

    op = MyOp()
    print(op([{'text': 'a'}]))
    ```
    """
        raise NotImplementedError()

    def forward_batch_input(self, inputs, **kwargs):
        """子类可实现的批量处理方法，用于在非逐条并发场景下直接接收整个输入列表并返回最终结果列表（可用于自定义批量逻辑或外部服务一次性处理）。

Args:
    inputs (list[dict]): 输入数据列表。
    **kwargs (dict): 其它用户传入的参数。


Examples:
    ```python
    from lazyllm.tools.data import LazyLLMDataBase

    class BatchOp(LazyLLMDataBase):
        def forward_batch_input(self, inputs):
            # implement batch processing and return a list
            return [{'text': i.get('text', '').lower()} for i in inputs]

    op = BatchOp()
    print(op([{'text': 'A'}, {'text': 'B'}]))
    ```
    """
        raise NotImplementedError()

    def _run_one(self, data):
        try:
            kwargs = getattr(self, '_lazyllm_kwargs', {})
            return self.forward(data, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if isinstance(data, dict):
                return {**data, 'infer_error': err_msg}
            return {'input': data, 'infer_error': err_msg}

    def _process_forward_common(self, data):
        self._store.load_progress()
        results = []
        pbar = tqdm(total=len(data), desc=f'Processing {self.__class__.__name__}', unit='item')

        if self._store.is_done:
            pbar.update(len(data))
            pending_indices = []
        else:
            if len(self._store.processed_indices) > 0:
                pbar.update(len(self._store.processed_indices))

            pending_indices = [idx for idx in range(len(data)) if idx not in self._store.processed_indices]

        if not pending_indices:
            pbar.close()
            return self._store.load_results()

        if self._concurrency_mode == 'single':
            for idx in pending_indices:
                res = self._run_one(data[idx])
                self._handle_result(res, data[idx], results, [idx])
                pbar.update(1)
        else:
            self._process_parallel(data, pending_indices, results, pbar)

        pbar.close()
        # Flush remaining
        if self._store.save_data:
            self._store.save_results([], force=True)  # Flush
            return self._store.load_results()
        return results

    def _process_parallel(self, data, pending_indices, results, pbar):

        executor_cls = ProcessPoolExecutor if self._concurrency_mode == 'process' else ThreadPoolExecutor
        idx_iter = iter(pending_indices)
        futures = {}

        with executor_cls(max_workers=self._max_workers) as executor:
            # 1. Submit initial batch
            for _ in range(self._max_workers):
                try:
                    idx = next(idx_iter)
                    fut = executor.submit(self._run_one, data[idx])
                    futures[fut] = idx
                except StopIteration:
                    break

            # 2. Loop
            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    idx = futures.pop(fut)
                    try:
                        res = fut.result()
                        self._handle_result(res, data[idx], results, [idx])
                    except Exception as e:
                        if not self._ignore_errors:
                            raise e
                        LOG.error(f'Task failed: {e}')

                    pbar.update(1)

                    # Submit next
                    try:
                        next_idx = next(idx_iter)
                        new_fut = executor.submit(self._run_one, data[next_idx])
                        futures[new_fut] = next_idx
                    except StopIteration:
                        pass

    def _handle_result(self, res, original_data, results, indices):
        if isinstance(res, dict) and 'infer_error' in res:
            if self._store.save_data:
                self._store.save_errors(res)
                self._store.save_results([], indices)
            return

        # Logic to interpret return value
        final_res = []
        if res is None:
            final_res.append(original_data)  # Keep original
        elif isinstance(res, list):
            if res:  # Not empty
                final_res.extend(res)
            # Empty list means delete (do nothing)
        elif isinstance(res, dict):
            final_res.append(res)
        else:
            # Treat unexpected return types as errors
            err_msg = f'Invalid return type {type(res)} from {self.__class__.__name__}, expect dict or list or None'
            LOG.error(err_msg)
            if isinstance(original_data, dict):
                error_res = original_data.copy()
                error_res['infer_error'] = err_msg
            else:
                error_res = {'input': original_data, 'infer_error': err_msg}

            if self._store.save_data:
                self._store.save_errors(error_res)
                self._store.save_results([], indices)
            return

        if self._store.save_data:
            self._store.save_results(final_res, indices)
        else:
            results.extend(final_res)

    def _export_file(self, result):
        if not self._export_path or result is None:
            return result

        path = self._export_path
        if not path.endswith('.jsonl'):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f'{self.__class__.__name__}.jsonl')
        else:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        abs_path = os.path.abspath(path)
        with open(abs_path, 'w', encoding='utf-8') as f:
            for item in result:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return abs_path

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        kwargs = getattr(self, '_lazyllm_kwargs', {})
        res = []

        if self._overwrote('forward_batch_input'):
            self._store.load_progress()
            if self._store.save_data and self._store.resume and self._store.is_done:
                LOG.warning(f'skip {self.__class__.__name__} and load data from {self._store.save_path}')
                res = self._store.load_results()
            else:
                res = self.forward_batch_input(inputs, **kwargs)

                if self._store.save_data and res is not None:
                    self._store.save_results(res if isinstance(res, list) else [res], indices='Done', force=True)

        elif self._overwrote('forward'):
            res = self._process_forward_common(inputs)
        else:
            raise RuntimeError('Must implement forward or forward_batch_input')

        return self._export_file(res)

data_register = lazyllm.Register(
    LazyLLMDataBase, ['forward', 'forward_batch_input'],
    allowed_parameter=['_concurrency_mode'])

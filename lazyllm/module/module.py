import os
import inspect
import traceback
from lazyllm import ThreadPoolExecutor

import lazyllm
from lazyllm import FlatList, Option, kwargs, globals, locals, colored_text, redis_client
from lazyllm.common import _register_trim_module, HandledException, _change_exception_type
from ..components.formatter.formatterbase import file_content_hash, transform_path
from ..flow import FlowBase, Pipeline, Parallel
from ..common.bind import _MetaBind
import uuid
from ..hook import LazyLLMHook, LazyLLMFuncHook
from lazyllm import FileSystemQueue, LOG
from contextlib import contextmanager
from typing import Optional, Union, Dict, List, Callable
import copy
from collections import defaultdict
import sqlite3
import pickle
import hashlib
from abc import ABC, abstractmethod
from filelock import FileLock


lazyllm.config.add('cache_dir', str, os.path.join(os.path.expanduser(lazyllm.config['home']), 'cache'), 'CACHE_DIR',
                   description='The default result cache directory for module to use(Read and Write).')
lazyllm.config.add('cache_strategy', str, 'memory', 'CACHE_STRATEGY',
                   description='The default cache strategy to use(memory, file, sqlite, redis).')
lazyllm.config.add('cache_mode', str, 'RW', 'CACHE_MODE', options=['RW', 'RO', 'WO', 'NONE'],
                   description='The default cache mode to use(Read and Write, Read Only, Write Only, None).')
redis_client = redis_client['module']


class CacheNotFoundError(Exception): pass
class ModuleExecutionError(HandledException): pass


_register_trim_module({'lazyllm.module.module': ['__call__', '_call_impl']})


class _CacheStorageStrategy(ABC):
    def __init__(self, cache: Optional[bool] = False):
        if cache:
            self._cache_dir = os.path.join(lazyllm.config['cache_dir'], 'module')
            os.makedirs(self._cache_dir, exist_ok=True)
            self._lock = FileLock(os.path.join(self._cache_dir, 'cache.lock'))
        else:
            self._lock = lambda: contextmanager(lambda: (yield))()

    @abstractmethod
    def get(self, key: str, hash_key: str): pass

    @abstractmethod
    def set(self, key: str, hash_key: str, value): pass

    def close(self): pass  # noqa B027


class _MemoryCacheStrategy(_CacheStorageStrategy):
    def __init__(self):
        super().__init__()
        self._cache = defaultdict(dict)

    def get(self, key: str, hash_key: str):
        if key not in self._cache or hash_key not in self._cache[key]:
            raise CacheNotFoundError(f'Cache not found for {key}')
        return self._cache[key][hash_key]

    def set(self, key: str, hash_key: str, value):
        self._cache[key][hash_key] = value

    def close(self):
        self._cache.clear()


class _FileCacheStrategy(_CacheStorageStrategy):
    def __init__(self):
        super().__init__(cache=True)
        self.cache_file = os.path.join(self._cache_dir, 'cache.dat')

    def _load_cache(self):
        if not os.path.exists(self.cache_file): return {}
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}

    def _save_cache(self, cache_data):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass

    def get(self, key: str, hash_key: str):
        with self._lock:
            cache_data = self._load_cache()
            if key not in cache_data or hash_key not in cache_data[key]:
                raise CacheNotFoundError(f'Cache not found for {key}')
            return cache_data[key][hash_key]

    def set(self, key: str, hash_key: str, value):
        with self._lock:
            cache_data = self._load_cache()
            if key not in cache_data:
                cache_data[key] = {}
            cache_data[key][hash_key] = value
            self._save_cache(cache_data)


class _SQLiteCacheStrategy(_CacheStorageStrategy):
    def __init__(self):
        super().__init__(cache=True)
        self.db_path = os.path.join(self._cache_dir, 'cache.db')
        self.conn = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT,
                hash_key TEXT,
                value BLOB,
                PRIMARY KEY (key, hash_key)
            )
        ''')
        self.conn.commit()

    def get(self, key: str, hash_key: str):
        with self._lock:
            cursor = self.conn.execute(
                'SELECT value FROM cache WHERE key = ? AND hash_key = ?',
                (key, hash_key)
            )
            row = cursor.fetchone()
            if row is None:
                raise CacheNotFoundError(f'Cache not found for {key}')
            try:
                return pickle.loads(row[0])
            except Exception as e:
                raise CacheNotFoundError(f'Failed to deserialize cache for {key}: {e}')

    def set(self, key: str, hash_key: str, value):
        with self._lock:
            try:
                serialized_value = pickle.dumps(value)
                self.conn.execute(
                    'INSERT OR REPLACE INTO cache (key, hash_key, value) VALUES (?, ?, ?)',
                    (key, hash_key, serialized_value)
                )
                self.conn.commit()
            except Exception:
                pass

    def close(self):
        if self.conn:
            self.conn.close()


class _RedisCacheStrategy(_CacheStorageStrategy):
    def __init__(self, prefix: str = ''):
        if not redis_client: raise RuntimeError('Redis url should be set by `export LAZYLLM_REDIS_URL = xxx`')
        self._client = redis_client[prefix] if prefix else redis_client
        super().__init__()

    def _get_redis_key(self, key: str, hash_key: str):
        return f'{key}:{hash_key}'

    def get(self, key: str, hash_key: str):
        redis_key = self._get_redis_key(key, hash_key)
        value = self._client.get(redis_key)
        if value is None:
            raise CacheNotFoundError(f'Cache not found for {key}')
        try:
            return pickle.loads(value)
        except Exception as e:
            raise RuntimeError(f'Failed to deserialize cache for {key}: {e}')

    def set(self, key: str, hash_key: str, value):
        try:
            redis_key = self._get_redis_key(key, hash_key)
            serialized_value = pickle.dumps(value)
            self._client.set(redis_key, serialized_value)
        except Exception:
            pass


class ModuleCache(object):
    """模块缓存管理器，提供统一的缓存存储和检索功能。  
该类封装了多种缓存策略（内存、文件、SQLite、Redis），支持根据配置自动选择缓存存储方式，为模块执行结果提供高效的缓存机制。

功能特性:
    - 支持多种缓存策略：内存缓存、文件缓存、SQLite数据库缓存、Redis缓存。
    - 自动根据配置选择缓存策略，默认为内存缓存。
    - 支持缓存模式控制（读写、只读、只写、禁用）。
    - 提供统一的缓存接口，隐藏底层存储实现细节。
    - 支持参数哈希化，确保缓存键的唯一性。

Args:
    strategy (Optional[str]): 缓存策略，可选值为 'memory'、'file'、'sqlite'、'redis'。默认为 None，将使用配置中的策略。

使用场景:
    1. 为模块执行结果提供缓存，避免重复计算。
    2. 在分布式环境中使用 Redis 缓存实现共享。
    3. 使用文件或数据库缓存实现持久化存储。
    4. 根据性能需求选择不同的缓存策略。
"""
    def __init__(self, strategy: Optional[str] = None):
        self._strategy = self._create_strategy(strategy or lazyllm.config['cache_strategy'])

    def _create_strategy(self, strategy: str) -> _CacheStorageStrategy:
        strategy = strategy.lower()
        strategies = {
            'memory': _MemoryCacheStrategy,
            'file': _FileCacheStrategy,
            'sqlite': _SQLiteCacheStrategy,
            'redis': _RedisCacheStrategy,
        }

        if strategy not in strategies:
            raise ValueError(f'Unsupported cache strategy: {strategy}. '
                             f'Available strategies: {list(strategies.keys())}')
        return strategies[strategy]()

    def _hash(self, args, kw):
        def process_value(value, hash_obj):
            meta = ''
            if isinstance(value, (list, tuple, dict, set)):
                meta = str(type(value)) + str(len(value))
            if isinstance(value, str):
                hash_obj.update(str(file_content_hash(value)).encode())
            elif isinstance(value, set):
                hash_obj.update((meta + '>').encode())
                for item in sorted(value):
                    process_value(item, hash_obj)
                hash_obj.update(('<' + meta).encode())
            elif isinstance(value, (list, tuple)):
                hash_obj.update((meta + '>').encode())
                for item in value:
                    process_value(item, hash_obj)
                hash_obj.update(('<' + meta).encode())
            elif isinstance(value, dict):
                hash_obj.update((meta + '>').encode())
                for k, v in sorted(value.items()):
                    key_meta = 'key:' + str(type(k)) + str(k)
                    hash_obj.update(key_meta.encode())
                    process_value(v, hash_obj)
                hash_obj.update(('<' + meta).encode())
            else:
                value_meta = str(type(value)) + str(value)
                hash_obj.update(value_meta.encode())
        hash_obj = hashlib.md5()
        process_value(args, hash_obj)
        if kw:
            process_value(kw, hash_obj)
        return hash_obj.hexdigest()

    def get(self, key, args, kw):
        """从缓存中获取数据。

根据提供的键和参数从缓存中检索数据。如果缓存模式不允许读取或数据不存在，将抛出异常。

Args:
    key: 缓存键，用于标识缓存数据。
    args: 位置参数，用于生成缓存哈希键。
    kw: 关键字参数，用于生成缓存哈希键。

**Returns:**

- 任意类型：缓存中存储的数据。

**异常:** 

- CacheNotFoundError: 当缓存中不存在指定数据时抛出。
- RuntimeError: 当缓存模式设置为只写（WO）时抛出。
"""
        if 'R' not in lazyllm.config['cache_mode']:
            raise CacheNotFoundError('Cannot read cache due to `LAZYLLM_CACHE_MODE = WO`')
        hash_key = self._hash(args, kw)
        value = self._strategy.get(key, hash_key)
        return transform_path(value, mode='r2a')

    def set(self, key, args, kw, value):
        """将数据存储到缓存中。

根据提供的键和参数将数据存储到缓存中。如果缓存模式不允许写入，则直接返回不执行存储操作。

Args:
    key: 缓存键，用于标识缓存数据。
    args: 位置参数，用于生成缓存哈希键。
    kw: 关键字参数，用于生成缓存哈希键。
    value: 要存储的数据。

**注意:** 

- 如果缓存模式设置为只读（RO）或禁用（NONE），此方法将直接返回而不执行存储操作。
"""
        if 'W' not in lazyllm.config['cache_mode']: return
        hash_key = self._hash(args, kw)
        value = transform_path(value, mode='a2r')
        self._strategy.set(key, hash_key, value)

    def close(self):
        """关闭缓存存储策略。

释放缓存存储策略占用的资源，如关闭数据库连接、清理内存缓存等。调用此方法后，缓存将不再可用。

**注意:** 

- 调用此方法后，缓存实例将无法继续使用。
- 不同的缓存策略可能有不同的资源清理行为。
"""
        self._strategy.close()


module_cache = ModuleCache()


# use _MetaBind:
# if bind a ModuleBase: x, then hope: isinstance(x, ModuleBase)==True,
# example: ActionModule.submodules:: isinstance(x, ModuleBase) will add submodule.
class ModuleBase(metaclass=_MetaBind):
    """ModuleBase 是 LazyLLM 的核心基类，定义了所有模块的统一接口和基础能力。  
它抽象了模块的训练、部署、推理和评测逻辑，并提供了子模块管理、钩子注册、参数传递和递归更新等机制。  
用户自定义的模块需要继承 ModuleBase，并实现 ``forward`` 方法来定义具体的推理逻辑。  

功能特性:
    - 统一管理子模块 (submodules)，自动追踪被持有的 ModuleBase 实例。
    - 支持 Option 类型的超参数设置，方便网格搜索与自动调参。
    - 提供钩子 (hook) 机制，可在调用前后执行自定义逻辑。
    - 封装训练 (train)、服务部署 (server)、评测 (eval) 的更新流程。
    - 支持 evalset 的加载与自动并行推理评测。

Args:
    return_trace (bool): 是否将推理结果写入 trace 队列，用于调试和追踪。默认为 ``False``。

使用场景:
    1. 当你需要组合训练、部署、推理和评测中的部分或全部能力时，例如一个 Embedding 模型需要同时训练与推理。
    2. 当你希望通过根模块调用 ``start``、``update``、``eval`` 等方法，递归管理其持有的子模块。
    3. 当你希望用户参数从外层模块自动传递到内部实现（参考 WebModule）。
    4. 当你希望自定义模块支持参数网格搜索（参考 TrialModule）。


Examples:
    >>> import lazyllm
    >>> class Module(lazyllm.module.ModuleBase):
    ...     pass
    ... 
    >>> class Module2(lazyllm.module.ModuleBase):
    ...     def __init__(self):
    ...         super(__class__, self).__init__()
    ...         self.m = Module()
    ... 
    >>> m = Module2()
    >>> m.submodules
    [<Module type=Module>]
    >>> m.m3 = Module()
    >>> m.submodules
    [<Module type=Module>, <Module type=Module>]
    """
    builder_keys = []  # keys in builder support Option by default

    def __new__(cls, *args, **kw):
        sig = inspect.signature(cls.__init__)
        paras = sig.parameters
        values = list(paras.values())[1:]  # paras.value()[0] is self
        for i, p in enumerate(args):
            if isinstance(p, Option):
                ann = values[i].annotation
                assert ann == Option or (isinstance(ann, (tuple, list)) and Option in ann), \
                    f'{values[i].name} cannot accept Option'
        for k, v in kw.items():
            if isinstance(v, Option):
                ann = paras[k].annotation
                assert ann == Option or (isinstance(ann, (tuple, list)) and Option in ann), \
                    f'{k} cannot accept Option'
        return object.__new__(cls)

    def __init__(self, *, return_trace=False):
        self._submodules = []
        self._evalset = None
        self._return_trace = return_trace
        self.mode_list = ('train', 'server', 'eval')
        self._set_mid()
        self._used_by_moduleid = None
        self._module_name = None
        self._options = []
        self.eval_result = None
        self._use_cache: Union[bool, str] = False
        self._hooks = set()

    def __setattr__(self, name: str, value):
        if isinstance(value, ModuleBase):
            self._submodules.append(value)
        elif isinstance(value, Option):
            self._options.append(value)
        elif name.endswith('_args') and isinstance(value, dict):
            for v in value.values():
                if isinstance(v, Option):
                    self._options.append(v)
        return super().__setattr__(name, value)

    def __getattr__(self, key):
        def _setattr(v, *, _return_value=self, **kw):
            k = key[:-7] if key.endswith('_method') else key
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                kw.update(v[1])
                v = v[0]
            if len(kw) > 0:
                setattr(self, f'_{k}_args', kw)
            setattr(self, f'_{k}', v)
            if hasattr(self, f'_{k}_setter_hook'): getattr(self, f'_{k}_setter_hook')()
            return _return_value
        keys = self.__class__.builder_keys
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            return None
        elif key.startswith('_') and key.endswith('_args') and (key[1:-5] in keys or f'{key[1:-4]}method' in keys):
            return dict()
        raise AttributeError(f'{self.__class__} object has no attribute {key}')

    def __call__(self, *args, **kw):
        hook_objs = []
        for hook_type in self._hooks:
            if isinstance(hook_type, LazyLLMHook):
                hook_objs.append(copy.deepcopy(hook_type))
            elif isinstance(hook_type, type):
                assert issubclass(hook_type, LazyLLMHook), f'{hook_type} is not a subclass of LazyLLMHook'
                hook_objs.append(hook_type(self))
            hook_objs[-1].pre_hook(*args, **kw)
        try:
            kw.update(locals['global_parameters'].get(self._module_id, dict()))
            if (files := locals['lazyllm_files'].get(self._module_id)) is not None: kw['lazyllm_files'] = files
            if (history := locals['chat_history'].get(self._module_id)) is not None: kw['llm_chat_history'] = history

            r = (self._call_impl(**args[0], **kw)
                 if args and isinstance(args[0], kwargs) else self._call_impl(*args, **kw))
            if self._return_trace:
                lazyllm.FileSystemQueue.get_instance('lazy_trace').enqueue(str(r))
        except HandledException as e: raise e
        except Exception as e:
            LOG.error(f'An error occured in {self.__class__}' + (f' with name {self.name}' if self.name else
                      '') + f'. Args: `{args}`, Kwargs: `{kw}`')
            raise _change_exception_type(e, ModuleExecutionError) from None

        for hook_obj in hook_objs[::-1]:
            hook_obj.post_hook(r)
        for hook_obj in hook_objs:
            hook_obj.report()
        self._clear_usage()
        return r

    def _call_impl(self, *args, **kw):
        if self._use_cache and 'R' in lazyllm.config['cache_mode']:
            try:
                return module_cache.get(self.__cache_hash__, args, kw)
            except CacheNotFoundError:
                self._cache_miss_handler()
        r = self.forward(**args[0], **kw) if args and isinstance(args[0], kwargs) else self.forward(*args, **kw)
        if self._use_cache and 'W' in lazyllm.config['cache_mode']:
            module_cache.set(self.__cache_hash__, args, kw, r)
        return r

    def _stream_output(self, text: str, color: Optional[str] = None, *, cls: Optional[str] = None):
        (FileSystemQueue.get_instance(cls) if cls else FileSystemQueue()).enqueue(colored_text(text, color))
        return ''

    @contextmanager
    def stream_output(self, stream_output: Optional[Union[bool, Dict]] = None):
        """上下文管理器，用于在推理或执行过程中进行流式输出。  
当提供字典类型的 ``stream_output`` 时，可指定输出前缀和后缀，以及对应颜色。

Args:
    stream_output (Optional[Union[bool, Dict]]): 流式输出配置。

        - 如果为布尔值 True，则开启默认流式输出。
        - 如果为字典，可包含以下键：

            - 'prefix' (str): 输出前缀文本。
            - 'prefix_color' (str, optional): 前缀颜色。
            - 'suffix' (str): 输出后缀文本。
            - 'suffix_color' (str, optional): 后缀颜色。
"""
        if stream_output and isinstance(stream_output, dict) and (prefix := stream_output.get('prefix')):
            self._stream_output(prefix, stream_output.get('prefix_color'))
        yield
        if isinstance(stream_output, dict) and (suffix := stream_output.get('suffix')):
            self._stream_output(suffix, stream_output.get('suffix_color'))

    def used_by(self, module_id):
        """设置当前模块被哪个模块使用，用于标记模块的调用关系。  
可链式调用，返回模块自身。

Args:
    module_id (str): 调用该模块的上层模块的唯一 ID。

**Returns:**

- ModuleBase: 返回模块自身，用于链式调用。
"""
        self._used_by_moduleid = module_id
        return self

    def _clear_usage(self):
        globals['usage'].pop(self._module_id, None)

    # interfaces
    def forward(self, *args, **kw):
        """前向计算接口，需要子类实现。  
该方法定义了模块接收输入并返回输出的逻辑，是模块作为仿函数的核心函数。

Args:
    *args: 可变位置参数，子类可根据实际需求定义输入。
    **kw: 可变关键字参数，子类可根据实际需求定义输入。
"""
        raise NotImplementedError

    def register_hook(self, hook_type: Union[LazyLLMHook, Callable]):
        """注册一个钩子（Hook），在模块调用时执行特定逻辑。  
钩子需要继承自 ``LazyLLMHook``，可用于在模块前向计算前后添加自定义操作，例如日志记录或统计。

Args:
    hook_type (LazyLLMHook): 待注册的钩子对象。
"""
        if not isinstance(hook_type, type) and not isinstance(hook_type, LazyLLMHook) and callable(hook_type):
            hook_type = LazyLLMFuncHook(hook_type)
        if not isinstance(hook_type, LazyLLMHook):
            raise TypeError(f'Invalid hook type: {type(hook_type)}, '
                            'must be subclass or instance of LazyLLMHook, or callable function')
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        """注销已注册的钩子。  
如果钩子存在于模块中，将其移除，使其不再在模块调用时执行。

Args:
    hook_type (LazyLLMHook): 待注销的钩子对象。
"""
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        """清空模块中所有已注册的钩子。  
调用后模块将不再执行任何钩子逻辑。
"""
        self._hooks = set()

    def _get_train_tasks(self):
        """定义训练任务，该函数返回训练的pipeline，重写了此函数的子类可以在update阶段被训练/微调。


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def _get_train_tasks(self):
    ...         return lazyllm.pipeline(lambda : 1, lambda x: print(x))
    ... 
    >>> MyModule().update()
    1
    """
        return None

    def _get_deploy_tasks(self):
        """定义部署任务，该函数返回训练的pipeline，重写了此函数的子类可以在update/start阶段被部署。


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def _get_deploy_tasks(self):
    ...         return lazyllm.pipeline(lambda : 1, lambda x: print(x))
    ... 
    >>> MyModule().start()
    1
    """
        return None

    def _get_post_process_tasks(self): return None

    def _set_mid(self, mid=None):
        self._module_id = mid if mid else str(uuid.uuid4().hex)
        return self

    @property
    def name(self):
        return self._module_name

    @name.setter
    def name(self, name):
        self._module_name = name

    @property
    def submodules(self):
        return self._submodules

    def evalset(self, evalset, load_f=None, collect_f=lambda x: x):
        """为模块设置评测集（evaluation set）。  
模块在调用 ``update`` 或 ``eval`` 时会使用评测集进行推理，并将评测结果存储在 ``eval_result`` 变量中。  

Args:
    evalset (Union[list, str]): 评测数据列表，或者评测数据文件路径。
    load_f (Optional[Callable]): 当 ``evalset`` 为文件路径时，用于加载文件并返回列表的函数，默认为 None。
    collect_f (Callable): 对评测结果进行后处理的函数，默认为 ``lambda x: x``。


Examples:
    >>> import lazyllm
    >>> m = lazyllm.module.TrainableModule().deploy_method(lazyllm.deploy.dummy).finetune_method(lazyllm.finetune.dummy).trainset("").mode("finetune").prompt(None)
    >>> m.evalset([1, 2, 3])
    >>> m.update()
    INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
    >>> print(m.eval_result)
    ["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
    """
        if isinstance(evalset, str) and os.path.exists(evalset):
            with open(evalset) as f:
                assert callable(load_f)
                self._evalset = load_f(f)
        else:
            self._evalset = evalset
        self.eval_result_collet_f = collect_f

    # TODO: add lazyllm.eval
    def _get_eval_tasks(self):
        def set_result(x): self.eval_result = x

        def parallel_infer():
            with ThreadPoolExecutor(max_workers=200) as executor:
                results = list(executor.map(lambda item: self(**item)
                                            if isinstance(item, dict) else self(item), self._evalset))
            return results
        if self._evalset:
            return Pipeline(parallel_infer,
                            lambda x: self.eval_result_collet_f(x),
                            set_result)
        return None

    # update module(train or finetune),
    def _update(self, *, mode: Optional[Union[str, List[str]]] = None, recursive: bool = True):  # noqa C901
        if not mode: mode = list(self.mode_list)
        if type(mode) is not list: mode = [mode]
        for item in mode:
            assert item in self.mode_list, f'Cannot find {item} in mode list: {self.mode_list}'
        # dfs to get all train tasks
        train_tasks, deploy_tasks, eval_tasks, post_process_tasks = FlatList(), FlatList(), FlatList(), FlatList()
        stack, visited = [(self, iter(self.submodules if recursive else []))], set()
        while len(stack) > 0:
            try:
                top = next(stack[-1][1])
                stack.append((top, iter(top.submodules)))
            except StopIteration:
                top = stack.pop()[0]
                if top._module_id in visited: continue
                visited.add(top._module_id)
                if 'train' in mode: train_tasks.absorb(top._get_train_tasks())
                if 'server' in mode: deploy_tasks.absorb(top._get_deploy_tasks())
                if 'eval' in mode: eval_tasks.absorb(top._get_eval_tasks())
                post_process_tasks.absorb(top._get_post_process_tasks())

        if 'train' in mode and len(train_tasks) > 0:
            Parallel(*train_tasks).set_sync(True)()
        if 'server' in mode and len(deploy_tasks) > 0:
            if redis_client:
                Parallel(*deploy_tasks).set_sync(False)()
            else:
                Parallel.sequential(*deploy_tasks)()
        if 'eval' in mode and len(eval_tasks) > 0:
            Parallel.sequential(*eval_tasks)()
        Parallel.sequential(*post_process_tasks)()
        return self

    def update(self, *, recursive: bool = True):
        """更新模块（及所有的子模块）。当模块重写了 ``_get_train_tasks`` 方法后，模块会被更新。更新完后会自动进入部署和推理的流程。

Args:
    recursive (bool): 是否递归更新所有的子模块，默认为True


Examples:
    >>> import lazyllm
    >>> m = lazyllm.module.TrainableModule().finetune_method(lazyllm.finetune.dummy).trainset("").deploy_method(lazyllm.deploy.dummy).mode('finetune').prompt(None)
    >>> m.evalset([1, 2, 3])
    >>> m.update()
    INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
    >>> print(m.eval_result)
    ["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
    """
        return self._update(mode=['train', 'server', 'eval'], recursive=recursive)

    def update_server(self, *, recursive: bool = True):
        """更新模块及其子模块的部署（server）部分。当模块或子模块实现了部署功能时，会进行相应的服务启动。  

Args:
    recursive (bool): 是否递归更新所有子模块的部署任务，默认为 True。
"""
        return self._update(mode=['server'], recursive=recursive)

    def eval(self, *, recursive: bool = True):
        """对模块（及所有的子模块）进行评测。当模块通过 ``evalset`` 设置了评测集之后，本函数生效。

Args:
    recursive (bool): 是否递归评测所有的子模块，默认为True


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def forward(self, input):
    ...         return f'reply for input'
    ... 
    >>> m = MyModule()
    >>> m.evalset([1, 2, 3])
    >>> m.eval().eval_result
    ['reply for input', 'reply for input', 'reply for input']
    """
        return self._update(mode=['eval'], recursive=recursive)

    def start(self):
        """启动模块及所有子模块的部署服务。该方法会确保模块和子模块的 server 功能被执行，适合用于初始化或重新启动服务。

**Returns:**

- ModuleBase: 返回自身实例，以支持链式调用


Examples:
    >>> import lazyllm
    >>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
    >>> m.start()
    <Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
    >>> m(1)
    "reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
    """
        return self._update(mode=['server'], recursive=True)

    def restart(self):
        """重启模块及其子模块的部署服务。内部会调用 ``start`` 方法，实现服务的重新启动。

**Returns:**

- ModuleBase: 返回自身实例，以支持链式调用


Examples:
    >>> import lazyllm
    >>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
    >>> m.restart()
    <Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
    >>> m(1)
    "reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
    """
        return self.start()

    def wait(self):
        """等待模块或其子模块的执行完成。此方法在当前实现中为空，可由子类根据具体部署逻辑进行实现。
"""
        pass

    def stop(self):
        """停止模块及其所有子模块的运行。该方法会递归调用子模块的 ``stop`` 方法，适用于释放资源或关闭服务。
"""
        for m in self.submodules:
            m.stop()

    @property
    def options(self):
        options = self._options.copy()
        for m in self.submodules:
            options += m.options
        return options

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)

    def __repr__(self):
        return lazyllm.make_repr('Module', self.__class__, name=self.name)

    def for_each(self, filter, action):
        """对模块的所有子模块执行指定操作。递归遍历所有子模块，如果子模块满足 ``filter`` 条件，则执行 ``action``。

Args:
    filter (Callable): 接受子模块作为输入并返回布尔值的函数，用于判断是否执行操作。
    action (Callable): 对满足条件的子模块执行的操作函数。
"""
        for submodule in self.submodules:
            if filter(submodule):
                action(submodule)
            submodule.for_each(filter, action)

    @property
    def __cache_hash__(self):
        cache_hash = self.__class__.__name__
        if isinstance(self._use_cache, str): cache_hash += f'@{self._use_cache}'
        if hasattr(self, 'appendix_hash_key'): cache_hash += f'@{self.appendix_hash_key}'
        return cache_hash

    def use_cache(self, flag: Union[bool, str] = True):
        """启用或禁用模块的缓存功能。

此方法用于控制模块是否使用缓存来存储和检索执行结果，以提高性能并避免重复计算。

Args:
    flag (bool or str, optional): 缓存控制标志。如果为True，启用缓存；如果为False，禁用缓存；
                                 如果为字符串，使用特定的缓存标识符。默认为True。

**Returns:**

- 返回模块实例本身，支持方法链式调用。

"""
        self._use_cache = flag or False
        return self

    def _cache_miss_handler(self): pass


class ActionModule(ModuleBase):
    """用于将函数、模块、flow、Module等可调用的对象包装一个Module。被包装的Module（包括flow中的Module）都会变成该Module的submodule。

Args:
    action (Callable|list[Callable]): 被包装的对象，是一个或一组可执行的对象。
    return_trace (bool): 是否开启 trace 模式，用于记录调用栈，默认为 ``False``。
"""
    def __init__(self, *action, return_trace=False):
        super().__init__(return_trace=return_trace)
        if len(action) == 1 and isinstance(action, FlowBase): action = action[0]
        if isinstance(action, (tuple, list)):
            action = Pipeline(*action)
        assert isinstance(action, FlowBase), f'Invalid action type {type(action)}'
        self.action = action

    def forward(self, *args, **kw):
        """执行被包装的 action，对输入参数进行前向计算。等效于调用该模块本身。

Args:
    args (list of callables or single callable): 传递给被包装 action 的位置参数。
    kwargs (dict of callables): 传递给被包装 action 的关键字参数。

**Returns:**

- 任意类型：被包装 action 的执行结果。
"""
        return self.action(*args, **kw)

    @property
    def submodules(self):
        """返回被包装 action 中所有属于 ModuleBase 类型的子模块。该属性会自动展开 Pipeline 中嵌套的模块。

**Returns:**

- list[ModuleBase]: 子模块列表
"""
        try:
            if isinstance(self.action, FlowBase):
                submodule = []
                self.action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
                return submodule
        except Exception as e:
            raise RuntimeError(f'{str(e)}\nOriginal traceback:\n{"".join(traceback.format_tb(e.__traceback__))}')
        return super().submodules

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Action', subs=[repr(self.action)],
                                 name=self._module_name, return_trace=self._return_trace)


def flow_start(self):
    """启动流处理执行（已弃用）。

此方法已弃用，建议直接将流实例作为函数调用。执行流处理并返回结果。

Args:
    *args: 传递给流处理的可变位置参数。
    **kw: 传递给流处理的命名参数。

**Returns:**

- 流处理的结果。

**Note:**

- 此方法已标记为弃用，请使用流实例的直接调用方式代替。
"""
    ActionModule(self).start()
    return self


lazyllm.ReprRule.add_rule('Module', 'Action', 'Flow')
lazyllm.LazyLLMFlowsBase.start = flow_start


class ModuleRegistryBase(ModuleBase, metaclass=lazyllm.LazyLLMRegisterMetaClass):
    __reg_overwrite__ = 'forward'


register = lazyllm.Register(ModuleRegistryBase, ['forward'])

import re
import os
import copy
import json
import time
import types
import typing
import builtins
import tempfile
import threading
import functools
from abc import ABCMeta
from pathlib import Path
from ..configs import config
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import Any, Callable, Optional, List, Dict, Iterable

try:
    from typing import final
except ImportError:
    _F = typing.TypeVar('_F', bound=Callable[..., Any])

    def final(f: _F) -> _F:
        """A decorator to indicate final methods and final classes.

    Use this decorator to indicate to type checkers that the decorated
    method cannot be overridden, and decorated class cannot be subclassed.
    For example:

      class Base:
          @final
          def done(self) -> None:
              ...
      class Sub(Base):
          def done(self) -> None:  # Error reported by type checker
                ...

      @final
      class Leaf:
          ...
      class Other(Leaf):  # Error reported by type checker
          ...

    There is no runtime checking of these properties.
    """
        return f

try:
    from typing import override
except ImportError:
    def override(func: Callable):
        return func


class FlatList(list):
    def absorb(self, item):
        """添加元素到列表中。

Args:
    item: 要添加的元素，可以是单个元素或列表
"""
        if isinstance(item, list):
            self.extend(item)
        elif item is not None:
            self.append(item)


class ArgsDict(dict):
    """参数字典类，用于管理和验证命令行参数。

Args:
    *args: 传递给父类dict的 positional arguments
    **kwargs: 传递给父类dict的 keyword arguments

**Returns:**

- ArgsDict实例，提供参数检查和格式化功能
"""
    def __init__(self, *args, with_line=True, **kwargs):
        super(ArgsDict, self).__init__(*args, **kwargs)
        self._with_line = with_line

    def check_and_update(self, kw):
        """检查并更新参数字典。

Args:
    kw (dict): 要更新的参数字典
"""
        if not kw.pop('skip_check', config['deploy_skip_check_kw']):
            assert set(kw.keys()).issubset(set(self)), f'unexpected keys: {set(kw.keys()) - set(self)}'
        self.update(kw)

    def parse_kwargs(self):
        """将参数字典解析为命令行参数字符串。
"""
        string = []
        for k, v in self.items():
            if type(v) is dict:
                v = json.dumps(v).replace('\"', '\\\"')
            if self._with_line:
                string.append(f'--{k}={v}' if type(v) is not str else f'--{k}=\"{v}\"')
            else:
                string.append(f'{k}={v}' if type(v) is not str else f'{k}=\"{v}\"')
        string = ' '.join(string)
        return string

class CaseInsensitiveDict(dict):
    """大小写不敏感的字典类。

CaseInsensitiveDict 继承自 dict，提供大小写不敏感的键值存储和检索功能。所有的键都会被转换为小写形式存储，确保无论使用大写、小写或混合大小写的键名都能访问到相同的值。

特点：
    - 所有键在存储时自动转换为小写
    - 支持标准的字典操作（获取、设置、检查包含关系）
    - 保持字典的原有功能，只是键名处理方式不同

Args:
    *args: 传递给父类 dict 的位置参数
    **kwargs: 传递给父类 dict 的关键字参数


Examples:
    >>> from lazyllm.common import CaseInsensitiveDict
    >>> # 创建大小写不敏感的字典
    >>> d = CaseInsensitiveDict({'Name': 'John', 'AGE': 25, 'City': 'New York'})
    >>> 
    >>> # 使用不同大小写访问相同的键
    >>> print(d['name'])      # 使用小写
    ... 'John'
    >>> print(d['NAME'])      # 使用大写
    ... 'John'
    >>> print(d['Name'])      # 使用首字母大写
    ... 'John'
    >>> 
    >>> # 设置值时也会转换为小写
    >>> d['EMAIL'] = 'john@example.com'
    >>> print(d['email'])     # 使用小写访问
    ... 'john@example.com'
    >>> 
    >>> # 检查键是否存在（大小写不敏感）
    >>> 'AGE' in d
    True
    >>> 'age' in d
    True
    >>> 'Age' in d
    True
    >>> 
    >>> # 支持标准字典操作
    >>> d['PHONE'] = '123-456-7890'
    >>> print(d.get('phone'))
    ... '123-456-7890'
    >>> print(len(d))
    ... 5
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        for key, value in dict(*args, **kwargs).items():
            assert isinstance(key, str)
            self[key] = value

    def __getitem__(self, key):
        assert isinstance(key, str)
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        super().__setitem__(key.lower(), value)

    def __contains__(self, key):
        assert isinstance(key, str)
        return super().__contains__(key.lower())

# pack return value of modules used in pipeline / parallel.
# will unpack when passing it to the next item.
class package(tuple):
    """package类用于封装流水线或并行模块的返回值，保证传递给下游模块时自动拆包，从而支持多个值的灵活传递。


Examples:
    >>> from lazyllm.common import package
    >>> p = package(1, 2, 3)
    >>> p
    (1, 2, 3)
    >>> p[1]
    2
    >>> p_slice = p[1:]
    >>> isinstance(p_slice, package)
    True
    >>> p2 = package([4, 5])
    >>> p + p2
    (1, 2, 3, 4, 5)
    """
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, types.GeneratorType)):
            return super(__class__, cls).__new__(cls, args[0])
        else:
            return super(__class__, cls).__new__(cls, args)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return package(super(__class__, self).__getitem__(key))
        return super(__class__, self).__getitem__(key)

    def __add__(self, __other):
        return package(super().__add__(__other))


class kwargs(dict):
    pass


class arguments(object):
    class _None: pass

    def __init__(self, args=_None, kw=_None) -> None:
        self.args = package() if args is arguments._None else args
        if not isinstance(self.args, package): self.args = package((self.args,))
        self.kw = kwargs() if kw is arguments._None else copy.copy(kw)

    def append(self, x):
        args, kw = package(), kwargs()
        if isinstance(x, package):
            args = x
        elif isinstance(x, kwargs):
            kw = x
        elif isinstance(x, arguments):
            args, kw = x.args, x.kw
        else:
            args = package((x,))
        if args: self.args += args
        if kw:
            dup_keys = set(self.kw.keys()).intersection(set(kw.keys()))
            assert len(dup_keys) == 0, f'Duplicated keys: {dup_keys}'
            self.kw.update(kw)
        return self


builtins.package = package


class LazyLLMCMD(object):
    """命令行操作封装类，提供安全、灵活的命令行管理功能。

Args:
    cmd (Union[str, List[str], Callable]):命令行指令，支持三种形式：字符串命令,命令列表,可调用对象。
    return_value (Any):预设返回值。
    checkf(Any):命令验证函数。
    no_displays(Any):需要过滤的敏感参数名。


Examples:
    >>> from lazyllm.common import LazyLLMCMD
    >>> cmd = LazyLLMCMD("run --epochs=50 --batch-size=32")
    >>> print(cmd.get_args("epochs"))
    50
    >>> print(cmd.get_args("batch-size")) 
    32
    >>> base = LazyLLMCMD("python train.py", checkf=lambda x: True)
    >>> new = base.with_cmd("python predict.py")

    """
    def __init__(self, cmd, *, return_value=None, checkf=(lambda *a: True), no_displays=None):
        if isinstance(cmd, (tuple, list)):
            cmd = ' && '.join(cmd)
        assert isinstance(cmd, str) or callable(cmd), 'cmd must be func or (list of) bash command str.'
        self.cmd = cmd
        self.return_value = return_value
        self.checkf = checkf
        self.no_displays = no_displays

    def __hash__(self):
        return hash(self.cmd)

    def __str__(self):
        assert not callable(self.cmd), f'Cannot convert cmd function {self.cmd} to str'
        cmd = re.sub(r'\b(LAZYLLM_[A-Z0-9_]*?_(?:API|SECRET)_KEY)=\S+', r'\1=xxxxxx', self.cmd)
        if self.no_displays:
            for item in self.no_displays:
                pattern = r'(-{1,2}' + re.escape(item) + r')(\s|=|)(\S+|)'
                cmd = re.sub(pattern, '', cmd)
            return cmd
        else:
            return cmd

    def with_cmd(self, cmd):
        """创建新命令对象并继承当前配置。

Args:
    cmd: 新的命令内容（类型需与原始命令一致）

"""
        # Attention: Cannot use copy.deepcopy because of class method.
        new_instance = LazyLLMCMD(cmd, return_value=self.return_value,
                                  checkf=self.checkf, no_displays=self.no_displays)
        return new_instance

    def get_args(self, key):
        """从命令字符串中提取指定参数的值。

Args:
    key: 要提取的参数名
"""
        assert not callable(self.cmd), f'Cannot get args from function {self.cmd}'
        pattern = r'*(-{1,2}' + re.escape(key) + r')(\s|=|)(\S+|)*'
        return re.match(pattern, self.cmd)[3]

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(duration, *, msg=''):
    def raise_timeout_exception():
        event.set()

    event = threading.Event()
    timer = threading.Timer(duration, raise_timeout_exception)
    timer.start()

    try:
        yield
    finally:
        if not event.is_set():
            timer.cancel()
        else:
            raise TimeoutException(f'{msg}, block timed out after {duration} s')


class ReadOnlyWrapper(object):
    """
一个轻量级只读包装器，用于包裹任意对象并对外提供只读访问（实际并未完全禁止修改，但复制时不会携带原始对象）。包装器可以动态替换内部对象，并提供判断对象是否为空的辅助方法。

Args:
    obj (Optional[Any]): 初始被包装的对象，默认为 None。
"""
    def __init__(self, obj=None):
        self.obj = obj

    def set(self, obj):
        """
替换当前包装的内部对象。

Args:
    obj (Any): 新的内部对象。
"""
        self.obj = obj

    def __getattr__(self, key):
        # key will be 'obj' in copy.deepcopy
        if key != 'obj' and self.obj is not None:
            return getattr(self.obj, key)
        return super(__class__, self).__getattr__(key)

    # TODO: modify it
    def __repr__(self):
        r = self.obj.__repr__()
        return (f'{r[:-1]}' if r.endswith('>') else f'<{r}') + '(Readonly)>'

    def __deepcopy__(self, memo):
        # drop obj
        return ReadOnlyWrapper()

    def isNone(self):
        """
检查当前包装器是否未持有任何对象。

Args:
    None.

**Returns:**

- bool: 如果内部对象为 None 返回 True，否则 False。
"""
        return self.obj is None


class Identity():
    """
恒等模块，用于直接返回输入值。

该模块常用于模块拼接结构中占位，无实际处理逻辑。若输入为多个参数，将自动打包为一个整体结构输出。

Args:
    *args: 可选的位置参数，占位用。
    **kw: 可选的关键字参数，占位用。
"""
    def __init__(self, *args, **kw):
        pass

    def __call__(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return package(*inputs)

    def __repr__(self):
        return make_repr('Module', 'Identity')


class ResultCollector(object):
    """结果收集器，用于在流程或任务执行过程中按名称存储和访问结果。  
它通过调用自身（传入 name）返回一个可调用的 Impl 对象来收集指定名称的结果。  
适用于需要跨步骤共享中间结果的场景。
"""
    class Impl(object):
        def __init__(self, name, value): self._name, self._value = name, value

        def __call__(self, *args, **kw):
            assert (len(args) == 0) ^ (len(kw) == 0), f'args({len(args)}), kwargs({len(kw)})'
            assert self._name is not None
            if len(args) > 0:
                self._value[self._name] = args[0] if len(args) == 1 else package(*args)
                return self._value[self._name]
            else:
                self._value[self._name] = kw
                return kwargs(kw)

    def __init__(self): self._value = dict()

    def __call__(self, name): return ResultCollector.Impl(name, self._value)

    def __getitem__(self, name): return self._value[name]

    def __repr__(self): return repr(self._value)

    def keys(self):
        """获取所有已存储结果的名称。

**Returns:**

- KeysView[str]: 结果名称集合。
"""
        return self._value.keys()

    def items(self):
        """获取所有已存储的 (名称, 值) 对。

**Returns:**

- ItemsView[str, Any]: 结果的键值对集合。
"""
        return self._value.items()


class ReprRule(object):
    rules = {}

    @classmethod
    def add_rule(cls, cate, type, subcate, subtype=None):
        if subtype:
            cls.rules[f'{cate}:{type}'] = f'<{subcate} type={subtype}'
        else:
            cls.rules[f'{cate}:{type}'] = f'<{subcate}'

    @classmethod
    def check_combine(cls, cate, type, subs):
        return f'{cate}:{type}' in cls.rules and subs.startswith(cls.rules[f'{cate}:{type}'])


def rreplace(s, old, new, count):
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]

def make_repr(category: str, type: str, *, name: Optional[str] = None,
              subs: Optional[List[str]] = None, attrs: Optional[Dict[str, Any]] = None, **kw):
    subs, attrs = subs or [], attrs or {}
    if len(kw) > 0:
        assert len(attrs) == 0, 'Cannot provide attrs and kwargs at the same time'
        attrs = kw

    if not config['repr_show_child']: subs = []

    if isinstance(type, builtins.type): type = type.__name__
    name = f' name={name}' if name else ''
    attrs = ' ' + ' '.join([f'{k}={v}' for k, v in attrs.items()]) if attrs else ''
    repr = f'<{category} type={type}{name}{attrs}>'

    if len(subs) == 1 and ReprRule.check_combine(category, type, subs[0]):
        if config['repr_ml']:
            sub_cate = re.split('>| ', subs[0][1:])[0]
            subs = rreplace(subs[0], f'</{sub_cate}>', f'</{category}>', 1)
        else:
            subs = subs[0]
        return repr[:-1] + f' sub-category={subs[1:]}'

    # ident
    sub_repr = []
    for idx, value in enumerate(subs):
        for i, v in enumerate(value.strip().split('\n')):
            if not config['repr_ml']:
                if idx != len(subs) - 1:
                    sub_repr.append(f' |- {v}' if i == 0 else f' |  {v}')
                else:
                    sub_repr.append(f' └- {v}' if i == 0 else f'    {v}')
            else:
                sub_repr.append(f'    {v}')
    if len(sub_repr) > 0: repr += ('\n' + '\n'.join(sub_repr) + '\n')
    if config['repr_ml']: repr += f'</{category}>'
    return repr


# if key is already in repr, then modify its value.
# if ket is not in repr, add key to repr with value.
# if value is None, remove key from repr.
def modify_repr(repr, key, value):
    # TODO: impl this function
    return repr


class once_flag(object):
    def __init__(self, reset_on_pickle=False):
        self._flag = False
        self._exc = None
        self._reset_on_pickle = reset_on_pickle
        self._lock = threading.RLock()
        self._ignore_reset = False

    def set(self, flag=True, ignore_reset=False):
        with self._lock:
            self._flag = flag
            self._ignore_reset = ignore_reset

    def set_exception(self, exc):
        self._exc = exc

    def reset(self):
        if not self._ignore_reset:
            self.set(False)

    def __bool__(self):
        return self._flag

    @classmethod
    def rebuild(cls, flag, reset_on_pickle):
        r = cls(reset_on_pickle)
        if not reset_on_pickle: r._flag = flag
        return r

    def __reduce__(self):
        return once_flag.rebuild, (self._flag, self._reset_on_pickle)

def call_once(flag: once_flag, func: Callable, *args, **kw):
    with flag._lock:
        if not flag:
            try:
                return func(*args, **kw)
            except Exception as e:
                flag.set_exception(e)
            finally:
                flag.set()
        if flag._exc:
            raise flag._exc
    return None

def once_wrapper(reset_on_pickle):
    flag = reset_on_pickle if isinstance(reset_on_pickle, bool) else False

    class Wrapper:
        class Impl:
            def __init__(self, func, instance):
                self._func, self._instance = func, instance
                flag_name = f'_lazyllm_{func.__name__}_once_flag'
                if instance and not hasattr(instance, flag_name): setattr(instance, flag_name, once_flag(flag))

            def __call__(self, *args, **kw):
                assert self._instance is not None, f'{self._func} can only be used as instance method'
                return call_once(self.flag, self._func, self._instance, *args, **kw)

            __doc__ = property(lambda self: self._func.__doc__)
            def __repr__(self): return repr(self._func)

            @__doc__.setter
            def __doc__(self, value): self._func.__doc__ = value

            @property
            def flag(self) -> once_flag:
                return getattr(self._instance, f'_lazyllm_{self._func.__name__}_once_flag')

        def __init__(self, func):
            self.__func__ = func

        def __get__(self, instance, _):
            return Wrapper.Impl(self.__func__, instance)

    return Wrapper if isinstance(reset_on_pickle, bool) else Wrapper(reset_on_pickle)


class DynamicDescriptor:
    """动态描述符类，用于创建支持实例和类级别调用的描述符。

Args:
    func (callable): 要包装的函数或方法
"""
    class Impl:
        def __init__(self, func, instance, owner):
            self._func, self._instance, self._owner = func, instance, owner

        def __call__(self, *args, **kw):
            return self._func(self._instance, *args, **kw) if self._instance else self._func(self._owner, *args, **kw)

        def __repr__(self): return repr(self._func)
        __doc__ = property(lambda self: self._func.__doc__)

        @__doc__.setter
        def __doc__(self, value): self._func.__doc__ = value

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, owner):
        return DynamicDescriptor.Impl(self.__func__, instance, owner)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def reset_on_pickle(*fields):
    def decorator(cls):
        original_getstate = cls.__getstate__ if hasattr(cls, '__getstate__') else lambda self: self.__dict__
        original_setstate = (cls.__setstate__ if hasattr(cls, '__setstate__') else
                             lambda self, state: self.__dict__.update(state))

        def __getstate__(self):
            state = original_getstate(self).copy()
            for field, *_ in fields:
                state[field] = None
            return state

        def __setstate__(self, state):
            original_setstate(self, state)
            for field in fields:
                field, field_type = field if isinstance(field, (tuple, list)) else (field, None)
                if field in state and state[field] is None and field_type is not None:
                    setattr(self, field, field_type() if field_type else None)

        cls.__getstate__ = __getstate__
        cls.__setstate__ = __setstate__
        return cls
    return decorator

class EnvVarContextManager:
    """环境变量上下文管理器，用于 在代码块执行期间临时设置环境变量，退出时自动恢复原始环境变量。

Args:
    env_vars_dict (dict): 需要临时设置的环境变量字典，值为 None 的变量将被忽略。
"""
    def __init__(self, env_vars_dict):
        self.env_vars_dict = {var: value for var, value in env_vars_dict.items() if value is not None}
        self.original_values = {}

    def __enter__(self):
        for var, value in self.env_vars_dict.items():
            if var in os.environ:
                self.original_values[var] = os.environ[var]
            os.environ[var] = value
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for var in self.env_vars_dict:
            if var in self.original_values:
                os.environ[var] = self.original_values[var]
            else:
                del os.environ[var]

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_valid_path(path):
    return os.path.isfile(path)

class Finalizer(object):
    """终结器类，用于管理资源的清理和释放操作。可以作为上下文管理器使用，或通过对象销毁时自动触发清理。

Args:
    func1 (Callable): 主要的清理函数。如果提供了func2，则func1会立即执行，func2作为清理函数。
    func2 (Optional[Callable]): 可选的清理函数，默认为None。
    condition (Callable): 条件函数，返回True时才执行清理函数，默认总是返回True。

用途：
1. 可以作为上下文管理器使用（with语句）
2. 可以通过对象销毁时自动触发清理
3. 支持条件性清理
4. 支持两阶段初始化和清理

注意：
    - 当提供func2时，func1会在初始化时立即执行
    - 清理函数只会执行一次
    - 清理操作会在对象销毁或退出上下文时执行
"""
    def __init__(self, func1: Callable, func2: Optional[Callable] = None, *, condition: Callable = lambda: True):
        if func2:
            func1()
            func1 = func2
        self._func = func1
        self._condition = condition

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        if self._func:
            if self._condition(): self._func()
            self._func = None

class SingletonMeta(type):
    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonABCMeta(SingletonMeta, ABCMeta): pass


class TempPathGenerator(object):
    """
一个临时文件路径生成器，用于将字符串内容写入临时文件，并返回对应的文件路径列表。

该类实现了上下文管理协议（with 语法），在进入上下文时会为每一段文本创建一个临时文件，
并将其内容写入文件中；在退出上下文时，如果未设置 persist=True，则会自动清理临时目录。

常用于需要将内存中的文本内容临时转为文件路径，以复用现有基于文件路径的处理逻辑
（如文档加载、embedding、检索等）的场景。

Args:
    contents (Iterable[str]): 需要写入临时文件的文本内容，可以是字符串或字符串列表。
    suffix (str): 临时文件的后缀名，默认为 '.txt'。
    encoding (str): 写入文件时使用的编码格式，默认为 'utf-8'。
    persist (bool): 是否在退出上下文后保留临时文件。默认为 False，表示自动清理。


Examples:

    from lazyllm import TempPathGenerator

    texts = [
        "This is the first temporary document.",
        "This is the second temporary document."
    ]

    with TempPathGenerator(texts, suffix=".txt") as paths:
        for p in paths:
            print(p)
            # p can be passed to any file-based document loader
    """
    def __init__(self, contents: Iterable[str], *, suffix: str = '.txt', encoding: str = 'utf-8', persist: bool = False):
        if isinstance(contents, str): contents = [contents]
        self._contents, self._suffix, self._encoding, self._persist = list(contents), suffix, encoding, persist
        self._tmpdir = None
        self._paths: List[str] = []

    def __enter__(self) -> List[str]:
        if self._tmpdir: raise RuntimeError('TempPathGenerator is not thread-safe, please create a new object.')
        self._tmpdir = tempfile.TemporaryDirectory()
        base = Path(self._tmpdir.name)

        self._paths = []
        for i, text in enumerate(self._contents):
            p = base / f'{i}{self._suffix}'
            p.write_text(text, encoding=self._encoding)
            self._paths.append(str(p))
        return self._paths

    def __exit__(self, exc_type, exc, tb):
        if not self._persist:
            self._tmpdir.cleanup()


def retry(func: Optional[Callable] = None, *, stop_after_attempt: Optional[int] = None, delay: float = 0.0):
    """
一个简单的重试装饰器，用于在函数执行失败时按指定次数进行重试。

该装饰器会在被装饰函数抛出异常时捕获异常，并在未达到最大重试次数前
重新执行函数；如果超过最大重试次数仍然失败，则会抛出最后一次异常。
可选地支持在每次重试之间添加固定延迟。

适用于对不稳定操作（如网络请求、临时资源访问等）进行基础容错处理，
无需引入额外第三方依赖。

Args:
    stop_after_attempt (int): 最大重试次数，包含首次执行在内，默认为 3 次。
    delay (float): 每次重试之间的等待时间（秒），默认为 0，表示不等待。


Examples:

    import random
    from lazyllm import retry

    # default stop_after_attempt is 3
    @retry
    def unstable_function():
        if random.random() < 0.7:
            raise RuntimeError("Random failure")
        return "Success!"

    @retry(stop_after_attempt=3, delay=1.0)
    def unstable_function():
        if random.random() < 0.7:
            raise RuntimeError("Random failure")
        return "Success!"

    @retry(3)
    def unstable_function():
        if random.random() < 0.7:
            raise RuntimeError("Random failure")
        return "Success!"

    result = unstable_function()
    print(result)
    """
    if isinstance(func, int):
        assert stop_after_attempt is None
        stop_after_attempt = func
    stop_after_attempt = stop_after_attempt or 3

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, stop_after_attempt + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt >= stop_after_attempt:
                        raise
                    if delay > 0:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator(func) if callable(func) else decorator

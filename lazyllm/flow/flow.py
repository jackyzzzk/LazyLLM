import lazyllm
import builtins
from lazyllm import config
from lazyllm.common import LazyLLMRegisterMetaClass, package, kwargs, arguments, bind
from lazyllm.common import ReadOnlyWrapper, LOG, globals, locals, _get_callsite
from lazyllm.common import _register_trim_module, HandledException, _change_exception_type
from lazyllm.common.bind import _MetaBind
from functools import partial
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from asyncio import events
import types
import inspect
import threading
import traceback
import sys
import os
from typing import Union, List, Optional
import concurrent.futures
from collections import deque
import uuid
from ..hook import LazyLLMHook
from itertools import repeat


class FlowException(HandledException):
    pass


class _FuncWrap(object):
    def __init__(self, f):
        self._f = f._f if isinstance(f, _FuncWrap) else f

    def __call__(self, *args, **kw): return self._f(*args, **kw)

    def __repr__(self):
        # TODO: specify lambda/staticmethod/classmethod/instancemethod
        # TODO: add registry message
        return lazyllm.make_repr('Function', type(self._f).__name__, name=(
            self._f if _is_function(self._f) else self._f.__class__).__name__.strip('<>'))

    def __getattr__(self, __key):
        if __key != '_f':
            return getattr(self._f, __key)
        return super(__class__, self).__getattr__(__key)

_oldins = isinstance
def new_ins(obj, cls):
    if _oldins(obj, _FuncWrap) and os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) != 'ON':
        return True if (cls is _FuncWrap or (_oldins(cls, (tuple, list)) and _FuncWrap in cls)) else _oldins(obj._f, cls)
    return _oldins(obj, cls)

builtins.isinstance = new_ins

def _is_function(f):
    return isinstance(f, (types.BuiltinFunctionType, types.FunctionType,
                          types.BuiltinMethodType, types.MethodType, types.LambdaType))


_thread_local = threading.local()
_async_var = ContextVar('lazyllm.flow_stack')

def _get_flow_stack():
    if events._get_running_loop() is not None:
        stack = _async_var.get(None)
        if stack is None:
            stack = []
            _async_var.set(stack)
        return stack
    else:
        stack = getattr(_thread_local, 'stack', None)
        if stack is None:
            _thread_local.stack = stack = []
        return stack


_register_trim_module({'lazyllm.flow.flow': ['__call__']}, continuous=True)
_register_trim_module({'lazyllm.flow.flow': ['_run', 'invoke'], 'lazyllm.common.bind': ['__call__']})

class FlowBase(metaclass=_MetaBind):
    """用于构建流式结构的基类，可以容纳多个项目并组织成层次化结构。

该类允许将不同的对象（包括 ``FlowBase`` 实例或其他类型对象）组合在一起，
并为其分配可选的名称，从而支持按名称或索引访问。结构中的项目可以动态添加或遍历。

Args:
    *items: 要包含在流中的项目，可以是 ``FlowBase`` 的实例或其他对象。
    item_names (list of str, optional): 对应于每个项目的名称列表，会与 ``items`` 按顺序配对。
        如果未提供，所有项目的名称默认为 ``None``。
    auto_capture (bool, optional): 是否启用自动捕获。如果为 ``True``，在上下文管理器模式下，
        将自动捕获当前作用域中新定义的变量并加入流。默认为 ``False``。
"""
    def __init__(self, *items, item_names=None, auto_capture=False) -> None:
        self._father = None
        self._items, self._item_names, self._item_ids, self._item_pos = [], [], [], []
        self._auto_capture = auto_capture
        self._capture = True
        self._curr_frame = None
        self._flow_id = str(uuid.uuid4().hex)
        self._find_user_instantiation_frame()

        for k, v in zip(item_names if item_names else repeat(None), items):
            self._add(k, v)

        self._capture = False
        self._auto_registered = False

    def __post_init__(self): pass

    def _add(self, k, v):
        assert self._capture, f'_add can only be used in `{self.__class__}.__init__` or `with {self.__class__}()`'
        self._items.append(v() if isinstance(v, type) else _FuncWrap(v) if _is_function(v) or v in self._items else v)
        self._item_ids.append(k or str(uuid.uuid4().hex))
        self._item_pos.append(_get_callsite(depth=3))
        if isinstance(v, FlowBase): v._father = self
        if k:
            assert k not in self._item_names, f'Duplicated names {k}'
            self._item_names.append(k)
        if self._curr_frame and isinstance(v, FlowBase) and k and k not in self._curr_frame.f_locals:
            self._curr_frame.f_locals[k] = v  # make sense only when locals is globals

    def __enter__(self, __frame=None):
        assert len(self._items) == 0, f'Cannot init {self.__class__} with items if you want to use it by context.'
        self._curr_frame = __frame if __frame else inspect.currentframe().f_back
        if self._auto_capture:
            self._frame_keys = list(self._curr_frame.f_locals.keys())
        self._capture = True
        _get_flow_stack().append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_capture:
            locals = self._curr_frame.f_locals.copy()
            for var, val in locals.items():
                if var != 'self' and var not in self._frame_keys and (
                        (val._f if isinstance(val, bind) else val) is not self):
                    self._add(var, val)
        assert _get_flow_stack().pop() == self, 'Do not operate FLow.stack manually!'
        self._capture = False
        self._curr_frame = None
        self.__post_init__()
        return False

    def __iter__(self):
        # used to support `with pipeline() as (a.b, b):`
        return iter([self, self])

    def _find_user_instantiation_frame(self):
        try:
            for fm in inspect.stack():
                if fm.frame.f_globals.get('__name__', '').startswith('lazyllm.flow') or fm.filename.startswith('<'):
                    continue
                self._defined_file = os.path.abspath(fm.frame.f_code.co_filename)
                self._defined_func = fm.frame.f_code.co_name
                self._defined_pos = f'"file: {self._defined_file}", line {fm.frame.f_lineno}({self._defined_func})'
                break
        except Exception:
            self._defined_file, self._defined_pos, self._defined_func = None, None, None

    def _defined_at_the_same_scope(self, other: 'FlowBase'):
        return self._defined_file == other._defined_file and self._defined_func == other._defined_func

    def __setattr__(self, name: str, value):
        if '_capture' in self.__dict__ and self._capture and not name.startswith('_'):
            if len(_get_flow_stack()) > 1 and not self._auto_registered:
                super(__class__, self).__setattr__('_auto_registered', True)
                locals = self._curr_frame.f_locals.copy()
                for key, item in locals.items():
                    if item == self and (parent := _get_flow_stack()[-2]) != item:
                        if key not in parent._item_names and parent._defined_at_the_same_scope(self):
                            parent._add(key, self)
                        break
            assert name not in self._item_names, f'Duplicated name: {name}'
            self._add(name, value)
        elif name in getattr(self, '_item_names', ()):
            raise RuntimeError(f'The setting of {self.__class__} elements must be done within the `with` statement.')
        else:
            super(__class__, self).__setattr__(name, value)

    def __getattr__(self, name):
        if '_item_names' in self.__dict__ and name in self._item_names:
            return self._items[self._item_names.index(name)]
        raise AttributeError(f'{self.__class__} object has no attribute {name}')

    def id(self, module=None):
        """获取模块或流程的 ID。如果传入字符串则原样返回；如果传入已绑定的模块则返回其对应的 item_id；不传参时返回整个 flow 的唯一 id。

Args:
    module (Optional[Union[str, Any]]): 目标模块或字符串标识。

**Returns:**

- str: 对应的 ID 字符串。
"""
        if isinstance(module, str): return module
        return self._item_ids[self._items.index(module)] if module else self._flow_id

    @property
    def is_root(self):
        """一个属性，指示当前流项目是否是流结构的根。

**Returns:**

- bool: 如果当前项目没有父级（ ``_father`` 为None），则为True，否则为False。


Examples:
    >>> import lazyllm
    >>> p = lazyllm.pipeline()
    >>> p.is_root
    True
    >>> p2 = lazyllm.pipeline(p)
    >>> p.is_root
    False
    >>> p2.is_root
    True
    """
        return self._father is None

    @property
    def ancestor(self):
        """一个属性，返回当前流项目的最顶层祖先。

如果当前项目是根，则返回其自身。

**Returns:**

- FlowBase: 最顶层的祖先流项目。


Examples:
    >>> import lazyllm
    >>> p = lazyllm.pipeline()
    >>> p2 = lazyllm.pipeline(p)
    >>> p.ancestor is p2
    True
    """
        if self.is_root: return self
        return self._father.ancestor

    def for_each(self, filter, action):
        """对流中每个通过过滤器的项目执行一个操作。

该方法递归地遍历流结构，将操作应用于通过过滤器的每个项目。

Args:
    filter (callable): 一个接受项目作为输入并返回bool的函数，如果该项目应该应用操作，则返回True。
    action (callable): 一个接受项目作为输入并对其执行某些操作的函数。

**Returns:**

- None


Examples:
    >>> import lazyllm
    >>> def test1(): print('1')
    ... 
    >>> def test2(): print('2')
    ... 
    >>> def test3(): print('3')
    ... 
    >>> flow = lazyllm.pipeline(test1, lazyllm.pipeline(test2, test3))
    >>> flow.for_each(lambda x: callable(x), lambda x: print(x))
    <Function type=test1>
    <Function type=test2>
    <Function type=test3>
    """
        for item in self._items:
            if isinstance(item, FlowBase):
                item.for_each(filter, action)
            elif filter(item):
                action(item)

def _bind_enter(self):
    assert isinstance(self._f, FlowBase)
    self._f.__enter__() if type(self._f) is bind else self._f.__enter__(inspect.currentframe().f_back)
    return self

def _bind_exit(self, exc_type, exc_val, exc_tb):
    return (self._f.__exit__(exc_type, exc_val, exc_tb)
            if type(self._f) is bind else self._f.__exit__(exc_type, exc_val, exc_tb))

bind.__enter__ = _bind_enter
bind.__exit__ = _bind_exit


# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowsBase(FlowBase, metaclass=LazyLLMRegisterMetaClass):
    """一个支持流程封装、钩子注册与调用逻辑的基础类。

`LazyLLMFlowsBase` 是 LazyLLM 中所有流程（Flow）的基类，用于组织一系列可调用模块的执行流程，并支持钩子（hook）机制、同步控制、后处理逻辑等功能。它的设计旨在统一封装执行调用、异常处理、后处理、流程表示等功能，适用于各种同步数据处理场景。

该类通常不直接使用，而是被诸如 `Pipeline`、`Parallel` 等具体流程类继承和使用。

```text
输入 --> [Flow模块1 -> Flow模块2 -> ... -> Flow模块N] --> 输出
                   ↑             ↓
               pre_hook       post_hook
```

Args:
    args: 可变长度参数列表。
    post_action: 在主流程结束后对输出进行进一步处理的可调用对象。默认为 ``None``。
    auto_capture: 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
    **kw: 命名组件的键值对。
"""
    def __init__(self, *args, post_action=None, auto_capture=False, **kw):
        assert len(args) == 0 or len(kw) == 0, f'Cannot provide args `{args}` and kwargs `{kw}` at the same time'
        if len(args) > 0 and isinstance(args[0], (tuple, list)):
            assert len(args) == 1, 'args should be list of callable functions'
            args = args[0]
        args = list(args) + [v() if isinstance(v, type) else v for v in kw.values()]
        super(__class__, self).__init__(*args, item_names=list(kw.keys()), auto_capture=auto_capture)
        self.post_action = post_action() if isinstance(post_action, type) else post_action
        self._sync = False
        self._hooks = set()

    def __call__(self, *args, **kw):
        hook_objs = []
        for hook_type in self._hooks:
            if isinstance(hook_type, LazyLLMHook):
                hook_objs.append(hook_type)
            else:
                hook_objs.append(hook_type(self))
            hook_objs[-1].pre_hook(*args, **kw)
        output = self._run(args[0] if len(args) == 1 else package(args), **kw)
        if self.post_action is not None: self.invoke(self.post_action, output)
        if self._sync: self.wait()
        r = self._post_process(output)
        for hook_obj in hook_objs[::-1]:
            hook_obj.post_hook(r)
        for hook_obj in hook_objs:
            hook_obj.report()
        return r

    def register_hook(self, hook_type: LazyLLMHook):
        """注册一个 Hook 类型，用于在流程执行前后进行额外处理。

Args:
    hook_type (LazyLLMHook): 要注册的 Hook 类型或实例。
"""
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        """注销已注册的 Hook。

Args:
    hook_type (LazyLLMHook): 要移除的 Hook 类型或实例。
"""
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        """清空所有已注册的 Hook。
"""
        self._hooks = set()

    def _post_process(self, output):
        return output

    def _run(self, __input, **kw):
        raise NotImplementedError

    def start(self, *args, **kw):
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
        lazyllm.LOG.warning('start is depreciated, please use flow as a function instead')
        return self(*args, **kw)

    def set_sync(self, sync=True):
        """设置流程是否同步执行。

Args:
    sync (bool): 是否同步执行，默认为 True。

**Returns:**

- LazyLLMFlowsBase: 当前实例。
"""
        self._sync = sync
        return self

    def __repr__(self):
        subs = [repr(it) for it in self._items]
        if self.post_action is not None:
            subs.append(lazyllm.make_repr('Flow', 'PostAction', subs=[self.post_action.__repr__()]))
        return lazyllm.make_repr('Flow', self.__class__.__name__, subs=subs, items=self._item_names)

    def wait(self):
        """等待流程中所有异步任务完成。

**Returns:**

- LazyLLMFlowsBase: 当前实例。
"""
        def filter(x):
            return hasattr(x, 'job') and isinstance(x.job, ReadOnlyWrapper) and not x.job.isNone()
        self.for_each(filter, lambda x: x.job.wait())
        return self

    # bind_args: dict(input=input, args=dict(key=value))
    def invoke(self, it, __input, *, bind_args_source=None, **kw):
        """调用指定对象（可为函数、模块或 bind 对象）并传入输入数据。  
支持对 bind 对象进行 root/pipeline 输出替换。

Args:
    it (Callable | bind): 要调用的对象。
    __input (Any): 输入数据。
    bind_args_source (Any, optional): 绑定参数来源。
    **kw: 其他关键字参数。
"""
        if isinstance(it, bind):
            if isinstance(self, Pipeline):
                it._args = [self.output(a) if a in self._items else a for a in it._args]
                it._kw = {k: self.output(v) if v in self._items else v for k, v in it._kw.items()}
            kw['_bind_args_source'] = bind_args_source
        try:
            if not isinstance(it, LazyLLMFlowsBase) and isinstance(__input, (package, kwargs)):
                return it(*__input, **kw) if isinstance(__input, package) else it(**__input, **kw)
            else:
                return it(__input, **kw)
        except HandledException as e: raise e
        except Exception as e:
            try:
                pos = self._item_pos[self._items.index(it)]
            except Exception:
                pos = None
            if '_bind_args_source' in kw: kw = (kw.get('_bind_args_source') or {}).pop('kwargs', None)
            err_msg = (f'Flow defined at {self._defined_pos or "Unknown position"} encountered an error:\n'
                       f'invoking `{it}`({pos or "Position not found"}) with input `{__input}` and kw `{kw}` failed. '
                       + 'Details: `{type}: {value}`'.format(type=type(e).__name__, value=str(e).replace('\n', '\\n')))
            LOG.error(err_msg)
            LOG.debug(f'Error type: {type(e).__name__}, Error message: {str(e)}\n'
                      f'Traceback: {"".join(traceback.format_exception(*sys.exc_info()))}')
            raise _change_exception_type(e, FlowException) from None

    def bind(self, *args, **kw):
        """为当前流程绑定参数，生成一个 bind 对象。

Args:
    *args: 位置参数。
    **kw: 关键字参数。

**Returns:**

- bind: 绑定后的 bind 对象。
"""
        return bind(self, *args, **kw)


_async_save_flag = ContextVar('lazyllm.pipeline.save_flag')

def _get_current_save_flag():
    return (_async_save_flag.get(None) if events._get_running_loop() is not None else
            getattr(_thread_local, 'save_flag', None))

def _set_current_save_flag(flag):
    if events._get_running_loop() is not None:
        _async_save_flag.set(flag)
    else:
        _thread_local.save_flag = flag

@contextmanager
def save_pipeline_result(flag: bool = True):
    """一个上下文管理器，用于临时设置是否保存流水线中的中间执行结果。

在进入上下文时，会将 `Pipeline.g_save_flow_result` 设置为指定值；退出上下文后会恢复为原来的状态。适用于调试或需要记录中间输出的场景。

Args:
    flag (bool): 是否启用结果保存功能，默认为 True。

**Returns:**

- ContextManager: 上下文管理器。


Examples:
    >>> import lazyllm
    >>> pipe = lazyllm.pipeline(lambda x: x + 1, lambda x: x * 2)
    >>> with lazyllm.save_pipeline_result(True):
    ...     result = pipe(1)
    >>> result
    4
    """
    old_flag = _get_current_save_flag()
    try:
        _set_current_save_flag(flag)
        yield
    finally:
        _set_current_save_flag(old_flag)

# input -> module1 -> module2 -> ... -> moduleN -> output
#                                               \> post-action
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowsBase):
    """一个形成处理阶段管道的顺序执行模型。

 ``Pipeline``类是一个处理阶段的线性序列，其中一个阶段的输出成为下一个阶段的输入。它支持在最后一个阶段之后添加后续操作。它是 ``LazyLLMFlowsBase``的子类，提供了一个延迟执行模型，并允许以延迟方式包装和注册函数。

Args:
    args (list of callables or single callable): 管道的处理阶段。每个元素可以是一个可调用的函数或 ``LazyLLMFlowsBase.FuncWrap``的实例。如果提供了单个列表或元组，则将其解包为管道的阶段。
    post_action (callable, optional): 在管道的最后一个阶段之后执行的可选操作。默认为None。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs (dict of callables): 管道的命名处理阶段。每个键值对表示一个命名阶段，其中键是名称，值是可调用的阶段。

**Returns:**

- 管道的最后一个阶段的输出。


Examples:
    >>> import lazyllm
    >>> ppl = lazyllm.pipeline(
    ...     stage1=lambda x: x+1,
    ...     stage2=lambda x: f'get {x}'
    ... )
    >>> ppl(1)
    'get 2'
    >>> ppl.stage2
    <Function type=lambda>
    """
    def __init__(self, *args, post_action=None, auto_capture=False, save_result=None, **kw):
        super().__init__(*args, post_action=post_action, auto_capture=auto_capture, **kw)
        self._save_flow_result = save_result if save_result is not None else (
            _get_current_save_flag() or config['save_flow_result'])

    @property
    def save_flow_result(self): return self._save_flow_result

    @save_flow_result.setter
    def save_flow_result(self, value): self._save_flow_result = value

    @property
    def _loop_count(self):
        return getattr(self, '_loop_count_var', 1)

    @_loop_count.setter
    def _loop_count(self, count):
        assert count > 1, 'At least one loop is required!'
        self._loop_count_var = count

    @property
    def _stop_condition(self):
        return getattr(self, '_stop_condition_var', None)

    @_stop_condition.setter
    def _stop_condition(self, cond):
        self._stop_condition_var = cond

    @property
    def _judge_on_full_input(self):
        return getattr(self, '_judge_on_full_input_var', True)

    @_judge_on_full_input.setter
    def _judge_on_full_input(self, judge):
        self._judge_on_full_input_var = judge

    @property
    def input(self): return bind.Args(self.id())
    @property
    def kwargs(self): return bind.Args(self.id(), 'kwargs')

    def output(self, module, unpack=False):
        """获取流水线中指定模块的输出结果。

Args:
    module: 要获取输出的模块。可以是模块对象或模块名称。
    unpack (bool): 是否解包输出结果。默认为False。

**Returns:**

- bind.Args: 一个绑定参数对象，用于在流水线中传递数据。
"""
        return bind.Args(self.id(), self.id(module), unpack=unpack)

    def _run(self, __input, **kw):
        output = __input
        bind_args_source = dict(source=self.id(), input=output, kwargs=kw.copy())
        bind_flag = self.save_flow_result if (flag := _get_current_save_flag()) is None else flag
        if bind_flag:
            lazyllm.LOG.debug(f'add {self.id()} to bind_args')
            locals['bind_args'][self.id()] = bind_args_source
        for _ in range(self._loop_count):
            for it in self._items:
                output = self.invoke(it, output, bind_args_source=bind_args_source, **kw)
                kw.clear()
                bind_args_source[self.id(it)] = output
            exp = output
            if not self._judge_on_full_input:
                assert isinstance(output, tuple) and len(output) >= 2
                exp = output[0]
                output = output[1:]
            if callable(self._stop_condition) and self.invoke(self._stop_condition, exp): break
        if bind_flag:
            lazyllm.LOG.debug(f'delete {self.id()} form bind_args')
            locals['bind_args'].pop(self.id(), None)
        return output


config.add('save_flow_result', bool, False, 'SAVE_FLOW_RESULT',
           description='Whether to save the intermediate result of the pipeline.')


_barr = threading.local()
def barrier(args):
    if _barr.impl: _barr.impl.wait()
    return args


config.add('parallel_multiprocessing', bool, False, 'PARALLEL_MULTIPROCESSING',
           description='Whether to use multiprocessing for parallel execution, if not, default to use threading.')


#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowsBase):
    """用于管理LazyLLMFlows中的并行流的类。

这个类继承自LazyLLMFlowsBase，提供了一个并行或顺序运行操作的接口。它支持使用线程进行并发执行，并允许以字典形式返回结果。


可以这样可视化 ``Parallel`` 类：

```text
#       /> module11 -> ... -> module1N -> out1 \\
# input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#       \> module31 -> ... -> module3N -> out3 /
```        

可以这样可视化 ``Parallel.sequential`` 方法：

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: 基类的可变长度参数列表。
    _scatter (bool, optional): 如果为 ``True``，输入将在项目之间分割。如果为 ``False``，相同的输入将传递给所有项目。默认为 ``False``。
    _concurrent (bool, optional): 如果为 ``True``，操作将使用线程并发执行。如果为 ``False``，操作将顺序执行。默认为 ``True``。
    multiprocessing (bool, optional): 如果为 ``True``，将使用多进程而不是多线程进行并行执行。这可以提供真正的并行性，但会增加进程间通信的开销。默认为 ``False``。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs: 基类的任意关键字参数。

<span style="font-size: 20px;">&ensp;**`asdict property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为dict。当使用 ``asdict`` 时，请务必保证parallel的元素被取了名字，例如:  ``parallel(name=value)`` 。

<span style="font-size: 20px;">&ensp;**`astuple property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为tuple。

<span style="font-size: 20px;">&ensp;**`aslist property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为list。

<span style="font-size: 20px;">&ensp;**`sum property`**</span>

标记Parellel，使得Parallel每次调用时的返回值做一次累加。

<span style="font-size: 20px;">&ensp;**`join(self, string)`**</span>

标记Parellel，使得Parallel每次调用时的返回值通过 ``string`` 做一次join。


Examples:
    >>> import lazyllm
    >>> test1 = lambda a: a + 1
    >>> test2 = lambda a: a * 4
    >>> test3 = lambda a: a / 2
    >>> ppl = lazyllm.parallel(test1, test2, test3)
    >>> ppl(1)
    (2, 4, 0.5)
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3)
    >>> ppl(1)
    {2, 4, 0.5}
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).asdict
    >>> ppl(2)
    {'a': 3, 'b': 8, 'c': 1.0}
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).astuple
    >>> ppl(-1)
    (0, -4, -0.5)
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).aslist
    >>> ppl(0)
    [1, 0, 0.0]
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\\n')
    >>> ppl(1)
    '2\\n4\\n0.5'
    """

    @staticmethod
    def _worker(func, barrier, sid, local_data, *args, global_data=None, **kw):
        lazyllm.globals._init_sid(sid)
        if global_data: lazyllm.globals._update(global_data)
        lazyllm.locals._init_sid()
        lazyllm.locals._update(local_data)
        lazyllm.locals['bind_args'] = lazyllm.locals['bind_args'].copy()
        _barr.impl = barrier
        r = func(*args, **kw)
        lazyllm.locals.clear()
        return r

    class PostProcessType(Enum):
        NONE = 0
        DICT = 1
        TUPLE = 2
        LIST = 3
        SUM = 4
        JOIN = 5

    def __init__(self, *args, _scatter: bool = False, _concurrent: Union[bool, int] = True,
                 multiprocessing: bool = False, auto_capture: bool = False, **kw):
        super().__init__(*args, **kw, auto_capture=auto_capture)
        self._post_process_type = Parallel.PostProcessType.NONE
        self._post_process_args = None
        self._multiprocessing = multiprocessing or config['parallel_multiprocessing']
        self._concurrent = 0 if not _concurrent else 5 if isinstance(_concurrent, bool) else _concurrent
        self._scatter = _scatter

    @staticmethod
    def _set_status(self, type, args=None):
        assert self._post_process_type is Parallel.PostProcessType.NONE, 'Cannor set post process twice'
        self._post_process_type = type
        self._post_process_args = args
        return self

    asdict = property(partial(_set_status, type=PostProcessType.DICT))
    astuple = property(partial(_set_status, type=PostProcessType.TUPLE))
    aslist = property(partial(_set_status, type=PostProcessType.LIST))
    sum = property(partial(_set_status, type=PostProcessType.SUM))

    def join(self, string=''):
        """标记Parallel，使得每次调用时的返回值通过指定字符串连接。

Args:
    string (str): 用于连接结果的字符串。默认为空字符串。

**Returns:**

- Parallel: 返回当前 Parallel 实例，其结果将被字符串连接。

**示例:**

```python
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\n')
>>> ppl(1)
'2\n4\n0.5'
```
"""
        assert isinstance(string, str), 'argument of join shoule be str'
        return Parallel._set_status(self, type=Parallel.PostProcessType.JOIN, args=string)

    @classmethod
    def sequential(cls, *args, **kw):
        """创建一个顺序执行的Parallel实例。

这个类方法会将 ``_concurrent`` 设置为 ``False``，使得所有操作按顺序执行而不是并行执行。

可以这样可视化 ``Parallel.sequential`` 方法：

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: 传递给 Parallel 构造函数的可变长度参数列表。
    kwargs: 传递给 Parallel 构造函数的关键字参数。

**Returns:**

- Parallel: 一个新的顺序执行的 Parallel 实例。
"""
        return cls(*args, _concurrent=False, **kw)

    # items = [a, b, c, d, e], skip_items = [1, 3]/['b', 'd'] -> items = [a, c, e]
    # input = [0, 1, 2, 3, 4] -> [0, 2, 4]
    # input = [0, 2, 4] -> [0, 2, 4]
    # input = dict(a=0, b=1, c=2, d=3, e=4) -> [0, 2, 4]
    # input = dict(a=0, c=2, e=4) -> [0, 2, 4]
    def _split_input(self, inputs, items, _kept_idx):
        if self._scatter:
            if isinstance(inputs, dict):
                inputs = package(inputs[n] for n in ([n for i, n in enumerate(self._item_names or repeat(None))
                                                     if i in _kept_idx] if _kept_idx else self._item_names))
            else:
                if isinstance(inputs, package) and len(inputs) == 1: inputs = inputs[0]
                assert isinstance(inputs, (tuple, list)), (
                    f'Only tuple and list input can be split automatically, your input is {inputs} <{type(inputs)}>')
                if _kept_idx and len(inputs) != len(_kept_idx):
                    assert len(inputs) == len(self._items)
                    inputs = [inp for i, inp in enumerate(inputs) if i in _kept_idx]
        else:
            inputs = [inputs] * len(items)
        return inputs

    def _run(self, __input, __items=None, *, _kept_items: Optional[Union[int, str, List[Union[int, str]]]] = None,
             _skip_items: Optional[Union[int, str, List[Union[int, str]]]] = None, **kw):
        if (items := __items) is None: items = self._items
        if _kept_items:
            if _skip_items: raise RuntimeError('Cannot provide `_kept_items` and `_skip_items` at the same time!')
            _kept_idx = [i for i, (_, n) in enumerate(zip(self._items, self._item_names or repeat(None)))
                         if i in _kept_items or n in _kept_items]
        else:
            _kept_idx = ([i for i, (_, n) in enumerate(zip(self._items, self._item_names or repeat(None)))
                          if i not in _skip_items and n not in _skip_items] if _skip_items else None)
        if _kept_idx: items = [it for i, it in enumerate(self._items) if i in _kept_idx]
        inputs = self._split_input(__input, items, _kept_idx)

        if self._concurrent:
            if self._multiprocessing:
                barrier, executor = None, lazyllm.ProcessPoolExecutor
                kw['global_data'] = lazyllm.globals._data
            else:
                barrier, executor = threading.Barrier(len(items)), concurrent.futures.ThreadPoolExecutor

            with executor(max_workers=self._concurrent) as e:
                futures = [e.submit(partial(self._worker, self.invoke, barrier, lazyllm.globals._sid,
                                            lazyllm.locals._data, it, inp, **kw))
                           for it, inp in zip(items, inputs)]
                if (not_done := concurrent.futures.wait(futures).not_done):
                    error_msgs = []
                    for future in not_done:
                        if (exc := future.exception()) is not None:
                            if (tb := getattr(future, '_traceback', None)):
                                tb_str = ''.join(traceback.format_exception(type(exc), exc, tb))
                            else:
                                tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                            error_msgs.append(f'Future: {future}\n{tb_str}')
                        else:
                            error_msgs.append(f'Future: {future} not complete without exception。')
                    raise RuntimeError('Parallel execute failed!\n' + '\n'.join(error_msgs))
                return package([future.result() for future in futures])
        else:
            return package(self.invoke(it, inp, **kw) for it, inp in zip(items, inputs))

    def _post_process(self, output):
        if self._post_process_type == Parallel.PostProcessType.DICT:
            assert self._item_names, 'Item name should be set when you want to return dict.'
            output = {k: v for k, v in zip(self._item_names, output)}
        elif self._post_process_type == Parallel.PostProcessType.TUPLE:
            output = tuple(output)
        elif self._post_process_type == Parallel.PostProcessType.LIST:
            output = list(output)
        elif self._post_process_type == Parallel.PostProcessType.SUM:
            output = ''.join([str(i) for i in output]) if isinstance(output[0], str) else sum(output, type(output[0])())
        elif self._post_process_type == Parallel.PostProcessType.JOIN:
            output = self._post_process_args.join([str(i) for i in output])
        return output


#                  /> in1 -> module11 -> ... -> module1N -> out1 \
#  (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                  \> in3 -> module31 -> ... -> module3N -> out3 /
class Diverter(Parallel):
    """一个流分流器，将输入通过不同的模块以并行方式路由。

Diverter类是一种专门的并行处理形式，其中多个输入分别通过一系列模块并行处理。然后将输出聚合并作为元组返回。

当您拥有可以并行执行的不同数据处理管道，并希望在单个流构造中管理它们时，此类非常有用。

```text
#                 /> in1 -> module11 -> ... -> module1N -> out1 \\
# (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                 \> in3 -> module31 -> ... -> module3N -> out3 /
```                    

Args:
    args: 可变长度参数列表，代表并行执行的模块。
    _concurrent (bool, optional): 控制模块是否应并行执行的标志。默认为 ``True``。可用 ``Diverter.sequential`` 代替 ``Diverter`` 来设置此变量。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs: 代表额外模块的任意关键字参数，其中键是模块的名称。

.. property:: 
    asdict

    和 ``parallel.asdict`` 一样


Examples:
    >>> import lazyllm
    >>> div = lazyllm.diverter(lambda x: x+1, lambda x: x*2, lambda x: -x)
    >>> div(1, 2, 3)
    (2, 4, -3)
    >>> div = lazyllm.diverter(a=lambda x: x+1, b=lambda x: x*2, c=lambda x: -x).asdict
    >>> div(1, 2, 3)
    {'a': 2, 'b': 4, 'c': -3}
    >>> div(dict(c=3, b=2, a=1))
    {'a': 2, 'b': 4, 'c': -3}
    """
    def __init__(self, *args, _concurrent: Union[bool, int] = True, auto_capture: bool = False, **kw):
        super().__init__(*args, _scatter=True, _concurrent=_concurrent, auto_capture=auto_capture, **kw)


#                  /> in1 \                            /> out1 \
#  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                  \> in3 /                            \> out3 /
# Attention: Cannot be used in async tasks, ie: training and deploy
# TODO: add check for async tasks
class Warp(Parallel):
    """一个流形变器，将单个模块并行应用于多个输入。

Warp类设计用于将同一个处理模块应用于一组输入。它有效地将单个模块“形变”到输入上，使每个输入都并行处理。输出被收集并作为元组返回。需要注意的是，这个类不能用于异步任务，如训练和部署。

```text
#                 /> in1 \                            /> out1 \
# (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                 \> in3 /                            \> out3 /
```

Args:
    args: 可变长度参数列表，代表要应用于所有输入的单个模块。
    _scatter (bool): 是否以分片方式拆分输入，默认 False。
    _concurrent (bool | int): 是否启用并发执行，可设定最大并发数。默认启用并发。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs: 未来扩展的任意关键字参数。

注意:
    - 只允许一个函数在warp中。
    - Warp流不应用于异步任务，如训练和部署。


Examples:
    >>> import lazyllm
    >>> warp = lazyllm.warp(lambda x: x * 2)
    >>> warp(1, 2, 3, 4)
    (2, 4, 6, 8)
    >>> warp = lazyllm.warp(lazyllm.pipeline(lambda x: x * 2, lambda x: f'get {x}'))
    >>> warp(1, 2, 3, 4)
    ('get 2', 'get 4', 'get 6', 'get 8')

    >>> from lazyllm import package
    >>> warp1 = lazyllm.warp(lambda x, y: x * 2 + y)
    >>> print(warp1([package(1,2), package(10, 20)]))
    (4, 40)
    """
    def __init__(self, *args, _concurrent: Union[bool, int] = True, auto_capture: bool = False, **kw):
        super().__init__(*args, _scatter=True, _concurrent=_concurrent, auto_capture=auto_capture, **kw)
        if len(self._items) > 1: self._items = [Pipeline(*self._items)]

    def __post_init__(self):
        if len(self._items) > 1: self._items = [Pipeline(*self._items)]

    def _run(self, __input, **kw):
        assert 1 == len(self._items), 'Only one function is enabled in warp'
        assert '_skip_items' not in kw, '`skip_items` is not allowed in warp'
        assert '_kept_items' not in kw, '`_kept_items` is not allowed in warp'
        return super(__class__, self)._run(__input, self._items * len(package(__input)), **kw)

    @property
    def asdict(self): raise NotImplementedError

# switch(conversion(input)):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
class Switch(LazyLLMFlowsBase):
    """一个根据条件选择并执行流的控制流机制。

 ``Switch``类提供了一种根据表达式的值或条件的真实性选择不同流的方法。它类似于其他编程语言中找到的switch-case语句。

```text
# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
```   

Args:
    args: 可变长度参数列表，交替提供条件和对应的流或函数。条件可以是返回布尔值的可调用对象或与输入表达式进行比较的值。
    conversion (callable, optional): 在进行条件匹配之前，对判定表达式 ``exp`` 进行转换或预处理的函数。默认为 ``None``。
    post_action (callable, optional): 在执行选定流后要调用的函数。默认为 ``None``。
    judge_on_full_input(bool): 如果设置为 ``True`` ， 则通过 ``switch`` 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

Raises:
    TypeError: 如果提供的参数数量为奇数，或者如果第一个参数不是字典且条件没有成对提供。


Examples:
    >>> import lazyllm
    >>> def is_positive(x): return x > 0
    ...
    >>> def is_negative(x): return x < 0
    ...
    >>> switch = lazyllm.switch(is_positive, lambda x: 2 * x, is_negative, lambda x : -x, 'default', lambda x : '000', judge_on_full_input=True)
    >>>
    >>> switch(1)
    2
    >>> switch(0)
    '000'
    >>> switch(-4)
    4
    >>>
    >>> def is_1(x): return True if x == 1 else False
    ...
    >>> def is_2(x): return True if x == 2 else False
    ...
    >>> def is_3(x): return True if x == 3 else False
    ...
    >>> def t1(x): return 2 * x
    ...
    >>> def t2(x): return 3 * x
    ...
    >>> def t3(x): return x
    ...
    >>> with lazyllm.switch(judge_on_full_input=True) as sw:
    ...     sw.case[is_1::t1]
    ...     sw.case(is_2, t2)
    ...     sw.case[is_3, t3]
    ...
    >>> sw(1)
    2
    >>> sw(2)
    6
    >>> sw(3)
    3
    """
    # Switch({cond1: M1, cond2: M2, ..., condN: MN})
    # Switch(cond1, M1, cond2, M2, ..., condN, MN)
    def __init__(self, *args, conversion=None, post_action=None, judge_on_full_input=True):
        if len(args) == 1 and isinstance(args[0], dict):
            self.conds, items = list(args[0].keys()), list(args[0].values())
        else:
            self.conds, items = list(args[0::2]), args[1::2]
        super().__init__(*items, post_action=post_action)
        self._judge_on_full_input = judge_on_full_input
        self._set_conversion(conversion)

    def _set_conversion(self, conversion):
        self._conversion = conversion

    def _run(self, __input, **kw):
        exp = __input
        if not self._judge_on_full_input:
            assert isinstance(__input, tuple) and len(__input) >= 2
            exp = __input[0]
            __input = __input[1] if len(__input) == 2 else __input[1:]

        if self._conversion:
            exp = self._conversion(*exp) if isinstance(exp, package) else self._conversion(exp)

        for idx, cond in enumerate(self.conds):
            if (callable(cond) and self.invoke(cond, exp) is True) or (exp == cond) or (
                    exp == package((cond,))) or cond == 'default':
                return self.invoke(self._items[idx], __input, **kw)

    class Case:
        def __init__(self, m) -> None: self._m = m
        def __call__(self, cond, func): self._m._add_case(cond, func)

        def __getitem__(self, key):
            if isinstance(key, slice):
                if key.start:
                    if (callable(key.step) and key.stop is None):
                        return self._m._add_case(key.start, key.step)
                    elif (key.step is None and callable(key.stop)):
                        return self._m._add_case(key.start, key.stop)
            elif isinstance(key, tuple) and len(key) == 2 and callable(key[1]):
                return self._m._add_case(key[0], key[1])
            raise RuntimeError(f'Only [cond::func], [cond:func] or [cond, func] is allowed in case, but you give {key}')

    @property
    def case(self): return Switch.Case(self)

    def _add_case(self, case, func):
        self.conds.append(case)
        self._add(None, func)


# result = cond(input) ? tpath(input) : fpath(input)
class IFS(LazyLLMFlowsBase):
    """在LazyLLMFlows框架中实现If-Else功能。

IFS（If-Else Flow Structure）类设计用于根据给定条件的评估有条件地执行两个提供的路径之一（真路径或假路径）。执行选定路径后，可以应用可选的后续操作，并且如果指定，输入可以与输出一起返回。

Args:
    cond (callable): 一个接受输入并返回布尔值的可调用对象。它决定执行哪个路径。如果 ``cond(input)`` 评估为True，则执行 ``tpath`` ；否则，执行 ``fpath`` 。
    tpath (callable): 如果条件为True，则执行的路径。
    fpath (callable): 如果条件为False，则执行的路径。
    post_action (callable, optional): 执行选定路径后执行的可选可调用对象。可以用于进行清理或进一步处理。默认为None。

**Returns:**

- 执行路径的输出。


Examples:
    >>> import lazyllm
    >>> cond = lambda x: x > 0
    >>> tpath = lambda x: x * 2
    >>> fpath = lambda x: -x
    >>> ifs_flow = lazyllm.ifs(cond, tpath, fpath)
    >>> ifs_flow(10)
    20
    >>> ifs_flow(-5)
    5
    """
    def __init__(self, cond, tpath, fpath, post_action=None):
        super().__init__(cond, tpath, fpath, post_action=post_action)

    def _run(self, __input, **kw):
        cond, tpath, fpath = self._items
        try:
            flag = cond()
        except Exception:
            flag = cond if isinstance(cond, bool) else self.invoke(cond, __input, **kw)
        return self.invoke(tpath if flag else fpath, __input, **kw)


#  in(out) -> module1 -> ... -> moduleN -> exp, out -> out
#      ⬆----------------------------------------|
class Loop(Pipeline):
    """初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。

Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出提前退出循环。

Args:
    item (callable or list of callables): 将在循环中应用的函数或可调用对象。
    stop_condition (callable, optional): 一个函数，它接受循环中最后一个项目的输出作为输入并返回一个布尔值。如果返回 ``True``，循环将停止。如果为 ``None``，循环将继续直到达到 ``count``。默认为 ``None``。
    count (int, optional): 运行循环的最大迭代次数。默认为 ``sys.maxsize``。
    post_action (callable, optional): 循环结束后调用的函数。默认为 ``None``。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    judge_on_full_input (bool): 如果设置为 ``True`` ，则通过 ``stop_condition`` 的输入进行条件判断；否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

Raises:
    AssertionError: 如果提供的 ``stop_condition`` 既不是 ``callable`` 也不是 ``None``。


Examples:
    >>> import lazyllm
    >>> loop = lazyllm.loop(lambda x: x * 2, stop_condition=lambda x: x > 10, judge_on_full_input=True)
    >>> loop(1)
    16
    >>> loop(3)
    12
    >>>
    >>> with lazyllm.loop(stop_condition=lambda x: x > 10, judge_on_full_input=True) as lp:
    ...    lp.f1 = lambda x: x + 1
    ...    lp.f2 = lambda x: x * 2
    ...
    >>> lp(0)
    14
    """
    def __init__(self, *item, stop_condition=None, count=sys.maxsize, post_action=None,
                 auto_capture=False, judge_on_full_input=True, **kw):
        super().__init__(*item, post_action=post_action, auto_capture=auto_capture, **kw)
        assert callable(stop_condition) or stop_condition is None
        self._judge_on_full_input = judge_on_full_input
        self._stop_condition = stop_condition
        self._loop_count = count


class Graph(LazyLLMFlowsBase):
    """一个基于有向无环图（DAG）的复杂流控制结构。

Graph类允许您创建复杂的处理图，其中节点表示处理函数，边表示数据流。它支持拓扑排序来确保正确的执行顺序，并可以处理多输入和多输出的复杂依赖关系。

Graph类特别适用于需要复杂数据流和依赖管理的场景，如机器学习管道、数据处理工作流等。

Args:
    post_action (callable, optional): 在图执行完成后要调用的函数。默认为 ``None``。
    auto_capture (bool, optional): 是否自动捕获上下文中的变量。默认为 ``False``。
    kwargs: 代表命名节点和对应函数的任意关键字参数。

**Returns:**

- 图的最终输出结果。
"""

    start_node_name, end_node_name = '__start__', '__end__'

    class Node:
        def __init__(self, func, name, arg_names=None):
            self.func, self.name, self.arg_names = func, name, None
            self.inputs, self.outputs = dict(), []

        def __repr__(self): return lazyllm.make_repr('Flow', 'Node', name=self.name)

    def __init__(self, *, post_action=None, auto_capture=False, **kw):
        super(__class__, self).__init__(post_action=post_action, auto_capture=auto_capture, **kw)

    def __post_init__(self):
        self._nodes = {n: Graph.Node(f, n) for f, n in zip(self._items, self._item_names)}
        self._nodes[Graph.start_node_name] = Graph.Node(None, Graph.start_node_name)
        self._nodes[Graph.end_node_name] = Graph.Node(lazyllm.Identity(), Graph.end_node_name)
        self._in_degree = {node: 0 for node in self._nodes.values()}
        self._out_degree = {node: 0 for node in self._nodes.values()}
        self._sorted_nodes = None
        self._constants = []

    def set_node_arg_name(self, arg_names):
        """设置节点的参数名称。

此方法用于为图中的节点设置函数参数的名称，这对于多参数函数的正确调用很重要。

Args:
    arg_names (list): 参数名称的列表，与节点创建时的顺序对应。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.add = lambda a, b: a + b
    ...     g.multiply = lambda x, y: x * y
    >>> g.set_node_arg_name([['x', 'y'], ['a', 'b']])
    >>> g._nodes['add'].arg_names
    ['x', 'y']
    >>> g._nodes['multiply'].arg_names
    ['a', 'b']
    """
        for node_name, name in zip(self._item_names, arg_names):
            self._nodes[node_name].arg_names = name

    @property
    def start_node(self):
        """获取图的起始节点。

**Returns:**

- Node: 图的起始节点（__start__）对象。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.process = lambda x: x * 2
    >>> start = g.start_node
    >>> start.name
    '__start__'
    """
        return self._nodes[Graph.start_node_name]

    @property
    def end_node(self):
        """获取图的结束节点。

**Returns:**

- Node: 图的结束节点（__end__）对象。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.process = lambda x: x * 2
    >>> end = g.end_node
    >>> end.name
    '__end__'
    """
        return self._nodes[Graph.end_node_name]

    def add_edge(self, from_node, to_node, formatter=None):
        """在图中添加一条边，定义节点之间的数据流。

此方法用于定义图中节点之间的连接关系，指定数据如何从一个节点流向另一个节点。

Args:
    from_node (str or Node): 源节点的名称或Node对象。
    to_node (str or Node): 目标节点的名称或Node对象。
    formatter (callable, optional): 可选的格式化函数，用于在传递数据时进行转换。默认为 ``None``。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.node1 = lambda x: x * 2
    ...     g.node2 = lambda x: x + 1
    ...     g.node3 = lambda x, y: x + y
    >>> g.add_edge('__start__', 'node1')
    >>> g.add_edge('node1', 'node2')
    >>> g.add_edge('node3', '__end__')
    >>> g._nodes['node1'].outputs
    [<Flow type=Node name=node2>]
    >>> def double_input(data):
    ...     return data * 2
    >>> g.add_edge('node1', 'node3', formatter=double_input)
    >>> g._nodes['node3'].inputs
    {'node1': <function double_input at ...>}
    """
        if isinstance(from_node, (tuple, list)):
            return [self.add_edge(f, to_node, formatter) for f in from_node]
        if isinstance(to_node, (tuple, list)):
            return [self.add_edge(from_node, t, formatter) for t in to_node]

        if isinstance(from_node, str): from_node = self._nodes[from_node]
        if isinstance(to_node, str): to_node = self._nodes[to_node]
        from_node.outputs.append(to_node)
        assert from_node.name not in to_node.inputs, f'Duplicate edges from {from_node.name} to {to_node.name}'
        to_node.inputs[from_node.name] = formatter
        self._in_degree[to_node] += 1
        self._out_degree[from_node] += 1

    def add_const_edge(self, constant, to_node):
        """添加一个常量边，将固定值传递给指定节点。

此方法用于将常量值作为输入传递给图中的节点，无需从其他节点获取数据。

Args:
    constant: 要传递的常量值。
    to_node (str or Node): 目标节点的名称或Node对象。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.add = lambda x, y: x + y
    >>> g.add_const_edge(10, 'add')
    >>> g._constants
    [10]
    """
        if isinstance(to_node, (tuple, list)):
            return [self.add_const_edge(constant, t) for t in to_node]
        if isinstance(to_node, str): to_node = self._nodes[to_node]
        to_node.inputs[f'_lazyllm_constant_{len(self._constants)}'] = None
        self._constants.append(constant)

    def topological_sort(self):
        """执行拓扑排序，返回正确的节点执行顺序。

此方法使用Kahn算法对有向无环图进行拓扑排序，确保所有依赖关系都得到满足。

**Returns:**

- List[Node]: 按拓扑顺序排列的节点列表。

Raises:
- ValueError: 如果图中存在循环依赖。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.node1 = lambda x: x * 2
    ...     g.node2 = lambda x: x + 1
    ...     g.node3 = lambda x, y: x + y
    >>> g.add_edge('__start__', 'node1')
    >>> g.add_edge('node1', 'node2')
    >>> g.add_edge('node1', 'node3')
    >>> g.add_edge('node2', 'node3')
    >>> g.add_edge('node3', '__end__')
    >>> sorted_nodes = g.topological_sort()
    >>> [node.name for node in sorted_nodes]
    ['__start__', 'node1', 'node2', 'node3', '__end__']
    >>> g.add_edge('node3', 'node1')
    >>> try:
    ...     g.topological_sort()
    ... except ValueError as e:
    ...     print("检测到循环依赖")
    检测到循环依赖
    """
        in_degree = self._in_degree.copy()
        queue = deque([node for node in self._nodes.values() if in_degree[node] == 0])
        sorted_nodes: List[Graph.Node] = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for output_node in node.outputs:
                in_degree[output_node] -= 1
                if in_degree[output_node] == 0:
                    queue.append(output_node)

        if len(sorted_nodes) != len(self._nodes):
            raise ValueError('Graph has a cycle')

        return [n for n in sorted_nodes if (self._in_degree[n] > 0 or self._out_degree[n] > 0)]

    def _get_input(self, name, node, intermediate_results, futures):
        if name.startswith('_lazyllm_constant_'):
            return self._constants[int(name.removeprefix('_lazyllm_constant_'))]
        if name not in intermediate_results['values']:
            r = futures[name].result()
            with intermediate_results['lock']:
                if name not in intermediate_results['values']:
                    intermediate_results['values'][name] = r
        r = intermediate_results['values'][name]
        if isinstance(r, Exception): raise r
        if node.inputs[name]:
            if isinstance(r, arguments) and not ((len(r.args) == 0) ^ (len(r.kw) == 0)):
                raise RuntimeError('Only one of args and kwargs can be given with formatter.')
            r = node.inputs[name]((r.args or r.kw) if isinstance(r, arguments) else r)
        return r

    def compute_node(self, sid, node, intermediate_results, futures):
        """计算单个节点的输出结果。

此方法是图的内部方法，用于执行单个节点的计算，包括获取输入数据、应用格式化函数、调用节点函数等。

Args:
    sid: 会话ID。
    node (Node): 要计算的节点。
    intermediate_results (dict): 中间结果存储。
    futures (dict): 异步任务字典。

**Returns:**

- 节点的计算结果。


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.add = lambda x, y: x + y
    ...     g.multiply = lambda x: x * 2
    >>> g.add_edge('__start__', 'add')
    >>> g.add_const_edge(5, 'add')
    >>> g.add_edge('add', 'multiply')
    >>> g.add_edge('multiply', '__end__')
    >>> result = g(3)  # x=3, y=5 (常量)
    >>> result
    16
    """
        globals._init_sid(sid)

        kw = {}
        if len(node.inputs) == 1:
            input = self._get_input(list(node.inputs.keys())[0], node, intermediate_results, futures)
        else:
            # TODO(wangzhihong): add complex rules: mixture of package / support kwargs / ...
            inputs = package(self._get_input(input, node, intermediate_results, futures)
                             for input in node.inputs.keys())
            input = arguments()
            for inp in inputs:
                input.append(inp)

        if isinstance(input, arguments):
            kw = input.kw
            input = input.args

        if node.arg_names:
            if not isinstance(input, (list, tuple, package)): input = [input]
            kw.update({name: value for name, value in zip(node.arg_names, input)})
            input = package()

        return self.invoke(node.func, input, **kw)

    def _run(self, __input, **kw):
        if not self._sorted_nodes: self._sorted_nodes = self.topological_sort()
        intermediate_results = dict(lock=threading.Lock(), values={})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for node in self._sorted_nodes:
                if node.name == '__start__':
                    intermediate_results['values'][node.name] = (
                        arguments(__input, kw) if (__input and kw) else (kw or __input))
                else:
                    future = executor.submit(self.compute_node, globals._sid, node, intermediate_results, futures)
                    futures[node.name] = future

        return futures[Graph.end_node_name].result()

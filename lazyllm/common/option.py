from typing import Any
import copy
import multiprocessing
from .logger import LOG


class _OptionIterator(object):
    def __init__(self, m):
        self.m = m
        self.reset()

    def reset(self): self.m._idx = -1
    def __len__(self): return len(self.m._objs)
    def __deepcopy__(self, *args, **kw): return self

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        self.m._next()
        return self.m._obj


class Option(object):
    """
LazyLLM 提供的选项管理类，用于管理多个选项值并在它们之间进行迭代。此类主要用于参数网格搜索和超参数调优场景。

Args:
    *obj: 一个或多个选项值，可以是任意类型的对象。如果传入单个列表或元组，会自动展开。至少需要提供两个选项。

主要特性：

- **多选项管理**: 可以管理多个不同的选项值。
- **迭代支持**: 支持标准 Python 迭代协议，可以遍历所有选项。
- **当前值访问**: 可直接访问当前选中的选项值。
- **深度复制**: 支持获取当前选中选项值的深拷贝。
- **循环迭代**: 可在所有选项之间循环迭代。

**注意**: 此类主要用于 LazyLLM 内部的参数搜索和实验管理，尤其在 `TrialModule` 中进行参数网格搜索时。


Examples:
    >>> import lazyllm
    >>> from lazyllm.common.option import Option
    >>> learning_rates = Option(0.001, 0.01, 0.1)
    >>> print(f"当前学习率: {learning_rates}")
    当前学习率: <Option options="(0.001, 0.01, 0.1)" curr="0.001">
    >>> print(f"所有选项: {list(learning_rates)}")
    所有选项: [0.001, 0.01, 0.1]
    """
    def __init__(self, *obj):
        if len(obj) == 1 and isinstance(obj[0], (tuple, list)): obj = obj[0]
        assert isinstance(obj, (tuple, list)) and len(obj) > 1, 'More than one option shoule be given'
        self._objs = obj
        self._idx = 0
        self._obj = self._objs[self._idx]

    def _next(self):
        self._idx += 1
        if self._idx == len(self._objs):
            self._idx = 0
            raise StopIteration

    def __setattr__(self, __name: str, __value: Any) -> None:
        object.__setattr__(self, __name, __value)
        if __name == '_idx' and 0 <= self._idx < len(self._objs):
            self._obj = self._objs[self._idx]

    def __deepcopy__(self, *args, **kw):
        return copy.deepcopy(self._obj)

    def __iter__(self):
        return _OptionIterator(self)

    def __repr__(self):
        return f'<Option options="{self._objs}" curr="{self._obj}">'


def rebuild(x): return x
def reduce(x): return rebuild, (x._obj,)
multiprocessing.reducer.ForkingPickler.register(Option, reduce)


def OptionIter(list_of_options, suboption_func=lambda x: []):
    LOG.info('Options:', list_of_options)

    def impl(cur, remain):
        for r in cur:
            new_remain = remain + suboption_func(r)
            if len(new_remain) == 0:
                yield [r]
            else:
                for r2 in impl(new_remain[0], new_remain[1:]):
                    yield [r] + r2
    return impl(list_of_options[0], list_of_options[1:])

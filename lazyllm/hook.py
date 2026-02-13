from abc import ABC, abstractmethod
import inspect
import ast

class LazyLLMHook(ABC):
    """LazyLLM 提供的钩子系统抽象基类，用于在函数或方法执行前后插入自定义逻辑。

此类是一个抽象基类（ABC），定义了钩子系统的基本接口。通过继承此类并实现其抽象方法，可以创建自定义的钩子来监控、记录或修改函数执行过程。

Args:
    obj: 要监控的对象（通常是函数或方法）。此对象会被存储在钩子实例中，供其他方法使用。

**注意**: 此类是抽象基类，不能直接实例化。必须继承此类并实现所有抽象方法才能使用。
"""

    @abstractmethod
    def __init__(self, obj):
        pass

    @abstractmethod
    def pre_hook(self, *args, **kwargs):
        """前置钩子方法，在被监控函数执行前调用。

这是一个抽象方法，需要在子类中实现。

Args:
    *args: 传递给被监控函数的位置参数。
    **kwargs: 传递给被监控函数的关键字参数。
"""
        pass

    @abstractmethod
    def post_hook(self, output):
        """后置钩子方法，在被监控函数执行后调用。

这是一个抽象方法，需要在子类中实现。

Args:
    output: 被监控函数的返回值。
"""
        pass

    def report():  # This is not an abstract method, but it is required to be implemented in subclasses.
        """生成钩子的执行报告。

这是一个抽象方法，需要在子类中实现。
"""
        raise NotImplementedError


def _check_and_get_pre_assign_number(func):
    func_node = ast.parse(inspect.getsource(func)).body[0]

    yield_nodes = [n for n in ast.walk(func_node) if isinstance(n, ast.Yield)]
    yield_count = len(yield_nodes)
    if yield_count == 0: return
    elif yield_count > 1: raise ValueError('function can have at most one yield')

    left_count = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            if any(isinstance(sub, ast.Yield) for sub in ast.walk(node.value)):
                target = node.targets[0]
                left_count = len(target.elts) if isinstance(target, ast.Tuple) else 1
                if left_count > 1: raise ValueError('function can have at most one pre-assign')
                break
    return left_count


class LazyLLMFuncHook(LazyLLMHook):
    """函数作为钩子的辅助函数，如果函数中存在 yield 语句，则 yield 之前的语句作为 pre_hook，yield 之后的语句作为 post_hook。

Args:
    func: 作为钩子的函数。
"""
    def __init__(self, func):
        self._func = func
        self._isgeneratorfunction = inspect.isgeneratorfunction(func)
        if self._isgeneratorfunction:
            self._left_count = _check_and_get_pre_assign_number(func)

    def pre_hook(self, *args, **kwargs):
        if self._isgeneratorfunction:
            self._generator = self._func(*args, **kwargs)
            next(self._generator)
        else:
            self._func(*args, **kwargs)

    def post_hook(self, output):
        assert self._isgeneratorfunction, 'post_hook is only supported for generator functions'
        try:
            self._generator.send(output) if self._left_count == 1 else next(self._generator)
        except StopIteration: pass

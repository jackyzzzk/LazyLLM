import lazyllm
from lazyllm import LazyLLMRegisterMetaClass
from lazyllm import LazyLLMCMD, ReadOnlyWrapper
from lazyllm import launchers, LazyLLMLaunchersBase
from typing import Union

class ComponentBase(object, metaclass=LazyLLMRegisterMetaClass):
    """组件基类，提供统一的接口与基础实现，便于创建不同类型的组件。  
组件通过指定的 Launcher 来执行任务，支持自定义任务执行逻辑。

Args:
    launcher (LazyLLMLaunchersBase or type, optional): 组件使用的启动器实例或启动器类，默认为空启动器（empty）。


Examples:
    >>> from lazyllm.components.core import ComponentBase
    >>> class MyComponent(ComponentBase):
    ...     def apply(self, x):
    ...         return x * 2
    >>> comp = MyComponent()
    >>> comp.name = "ExampleComponent"
    >>> print(comp.name)
    ExampleComponent
    >>> result = comp(10)
    >>> print(result)
    20
    >>> print(comp.apply(5))
    10
    """
    def __init__(self, *, launcher=launchers.empty()):  # noqa B008
        self._llm_name = None
        self.job = ReadOnlyWrapper()
        if isinstance(launcher, LazyLLMLaunchersBase):
            self._launcher = launcher
        elif isinstance(launcher, type) and issubclass(launcher, LazyLLMLaunchersBase):
            self._launcher = launcher()
        else:
            raise RuntimeError('Invalid launcher given:', launcher)

    def apply():
        """组件执行的核心方法，需由子类实现。  
定义组件的具体业务逻辑或任务执行步骤。  

**注意:**  
调用组件时，如果子类重写了此方法，则会调用此方法执行任务。  
"""
        raise NotImplementedError('please implement function \'apply\'')

    def cmd(self, *args, **kw) -> Union[str, tuple, list]:
        """生成组件的执行命令，需由子类实现。  
返回的命令可以是字符串、元组或列表，表示具体执行任务的指令。  

**注意:**  
调用组件时，如果未重写 `apply` 方法，将通过此命令生成任务并由启动器执行。  
"""
        raise NotImplementedError('please implement function \'cmd\'')

    @property
    def name(self): return self._llm_name
    @name.setter
    def name(self, name): self._llm_name = name

    @property
    def launcher(self): return self._launcher

    def _get_job_with_cmd(self, *args, **kw):
        cmd = self.cmd(*args, **kw)
        cmd = cmd if isinstance(cmd, LazyLLMCMD) else LazyLLMCMD(cmd)
        return self._launcher.makejob(cmd=cmd)

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f) or \
            getattr(self.__class__, '__reg_overwrite__', None) == f

    def __call__(self, *args, **kw):
        if self._overwrote('apply'):
            assert not self._overwrote('cmd'), (
                'Cannot overwrite \'cmd\' and \'apply\' in the same class')
            assert isinstance(self._launcher, launchers.Empty), 'Please use EmptyLauncher instead.'
            return self._launcher.launch(self.apply, *args, **kw)
        else:
            job = self._get_job_with_cmd(*args, **kw)
            self.job.set(job)
            return self._launcher.launch(job)

    def __repr__(self):
        return lazyllm.make_repr('lazyllm.llm.' + self.__class__._lazy_llm_group,
                                 self.__class__.__name__, name=self.name)


register = lazyllm.Register(ComponentBase, ['apply', 'cmd'])

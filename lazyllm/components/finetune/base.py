from lazyllm import launchers
from ..core import ComponentBase


class LazyLLMFinetuneBase(ComponentBase):
    """LazyLLM微调基础组件类，继承自ComponentBase。

提供大语言模型微调的基础功能，支持远程启动器配置和模型路径管理。

Args:
    base_model (str): 基础模型路径或标识
    target_path (str): 微调后模型输出路径
    launcher (Launcher, optional): 任务启动器，默认为远程启动器
"""
    __reg_overwrite__ = 'cmd'

    def __init__(self, base_model, target_path, *, launcher=launchers.remote()):  # noqa B008
        super().__init__(launcher=launcher)
        self.base_model = base_model
        self.target_path = target_path
        self.merge_path = None

    def __call__(self, *args, **kw):
        super().__call__(*args, **kw)
        if self.merge_path:
            return self.merge_path
        else:
            return self.target_path


class DummyFinetune(LazyLLMFinetuneBase):
    """DummyFinetune 是 [LazyLLMFinetuneBase][lazyllm.components.LazyLLMFinetuneBase] 的子类，用于占位实现微调逻辑。
此类主要用于演示或测试目的，因为它不执行任何实际的微调操作。

Args:
    base_model: 字符串，指定基础模型的名称，默认为 'base'。
    target_path: 字符串，指定微调输出的目标路径，默认为 'target'。
    launcher: 启动器实例，用于执行命令。默认为 [launchers.remote()][lazyllm.launchers.remote]。
    **kw: 其他关键字参数，这些参数会被保存以供后续使用。

Returns:
    一个字符串，表示一个占位命令。该字符串包括初始化时传递的参数。


Examples:
    >>> from lazyllm.components import DummyFinetune
    >>> from lazyllm import launchers
    >>> # 创建一个 DummyFinetune 实例
    >>> finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='custom_value')
    >>> # 调用 cmd 方法生成占位命令
    >>> command = finetuner.cmd('--example-arg', key='value')
    >>> print(command)
    ... echo 'dummy finetune!, and init-args is {'custom_arg': 'custom_value'}'
    """
    def __init__(self, base_model='base', target_path='target', *, launcher=launchers.remote(), **kw):  # noqa B008
        super().__init__(base_model, target_path, launcher=launchers.empty)
        self.kw = kw

    def cmd(self, *args, **kw) -> str:
        """`cmd` 方法生成一个用于微调的占位命令字符串。此方法主要用于测试或演示目的。

Args:
    *args: 要包含在命令中的位置参数（在本实现中未使用）。
    **kw: 要包含在命令中的关键字参数（在本实现中未使用）。

Returns:
    一个字符串，表示一个占位命令。该字符串包括初始化时传递的关键字参数 (`**kw`)，存储在 `self.kw` 中。

Example:
    如果类初始化时使用 `custom_arg='value'`，调用 `cmd` 方法将返回：
    `"echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'"`


Examples:
    >>> from lazyllm.components import DummyFinetune
    >>> from lazyllm import launchers
    >>> # 创建一个 DummyFinetune 实例，并传递初始化参数
    >>> finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='value')
    >>> # 调用 cmd 方法生成占位命令
    >>> command = finetuner.cmd()
    >>> # 打印生成的占位命令
    >>> print(command)
    ... echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'
    """
        return f'echo \'dummy finetune!, and init-args is {self.kw}\''

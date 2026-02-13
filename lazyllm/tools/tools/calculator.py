from lazyllm.module import ModuleBase
from math import * # noqa. import math functions for expressions

class Calculator(ModuleBase):
    """
简单计算器模块，继承自ModuleBase。

提供数学表达式计算功能，支持基本的算术运算和数学函数。


Examples:

    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    """
    def __init__(self):
        super().__init__()

    def forward(self, exp: str, *args, **kwargs):
        """
计算用户输入的表达式的值。

Args:
    exp (str): 需要计算的表达式的值。必须符合 Python 计算表达式的语法。可使用 Python math 库中的数学函数。
    *args: 可变位置参数
    **kwargs: 可变关键字参数


Examples:

    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    result1 = calc.forward("2 + 3 * 4")
    print(f"2 + 3 * 4 = {result1}")
    """
        return eval(exp)

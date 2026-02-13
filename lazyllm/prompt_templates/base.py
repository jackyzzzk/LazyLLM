from abc import ABC
from string import Formatter
from pydantic import BaseModel


class BasePromptTemplate(BaseModel, ABC):

    @staticmethod
    def get_template_variables(template: str) -> list[str]:
        """从给定的模板字符串中提取所有占位符变量名。

使用 Python 内置的 string.Formatter 解析模板，识别占位符，
并返回一个按字母顺序排序的唯一变量名列表。

Args:
    template (str): 包含占位符的提示模板字符串

Returns:
    list[str]: 模板中所有占位符变量名的排序列表

Raises:
    ValueError: 当模板格式非法或解析失败时抛出异常
"""
        try:
            input_variables = {
                v for _, v, _, _ in Formatter().parse(template) if v is not None
            }
            return sorted(input_variables)
        except Exception as e:
            raise ValueError(f'Error getting template variables: {e}')

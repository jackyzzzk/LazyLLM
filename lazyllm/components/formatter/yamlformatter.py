import yaml
from .formatterbase import JsonLikeFormatter
import lazyllm


class YamlFormatter(JsonLikeFormatter):
    """用于从 YAML 格式的字符串中提取结构化信息的格式化器。

继承自 JsonLikeFormatter，通过内部方法将字符串解析为 Python 对象后使用类 JSON 的方式提取字段。

适合用于处理包含嵌套结构的 YAML 文本，并结合格式化表达式获取目标数据。



Examples:
    >>> from lazyllm.components.formatter import YamlFormatter
    >>> formatter = YamlFormatter("{name,age}")
    >>> msg = \"\"\" 
    ... name: Alice
    ... age: 30
    ... city: London
    ... \"\"\"
    >>> formatter(msg)
    {'name': 'Alice', 'age': 30}
    """
    def _load(self, msg: str):
        try:
            return yaml.load(msg, Loader=yaml.SafeLoader)
        except Exception as e:
            lazyllm.LOG.info(f'Error: {e}')
            return ''

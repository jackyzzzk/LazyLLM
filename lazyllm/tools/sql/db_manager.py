from enum import Enum, unique
from typing import List, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod
from lazyllm.module import ModuleBase


@unique
class DBStatus(Enum):
    """An enumeration."""
    SUCCESS = 0
    FAIL = 1


class DBResult(BaseModel):
    status: DBStatus = DBStatus.SUCCESS
    detail: str = 'Success'
    result: Union[List, None] = None

class CommonMeta(type(ABC), type(ModuleBase)):
    pass

class DBManager(ABC, ModuleBase, metaclass=CommonMeta):
    """数据库管理器的抽象基类。

该类定义了构建数据库连接器的通用接口，包括 `execute_query` 抽象方法和 `desc` 描述属性。

Args:
    db_type (str): 数据库类型标识符，例如 'mysql'、'mongodb'。


Examples:
    >>> from lazyllm.components import DBManager
    >>> class DummyDB(DBManager):
    ...     def __init__(self):
    ...         super().__init__(db_type="dummy")
    ...     def execute_query(self, statement):
    ...         return f"Executed: {statement}"
    ...     @property
    ...     def desc(self):
    ...         return "Dummy database for testing."
    >>> db = DummyDB()
    >>> print(db("SELECT * FROM test"))
    ... Executed: SELECT * FROM test
    """

    def __init__(self, db_type: str):
        ModuleBase.__init__(self)
        self._db_type = db_type
        self._desc = None

    @abstractmethod
    def execute_query(self, statement) -> str:
        """执行数据库查询语句的抽象方法。此方法需要由具体的数据库管理器子类实现，用于执行各种数据库操作。

Args:
    statement: 要执行的数据库查询语句，可以是 SQL 语句或其他数据库特定的查询语言

此方法的特点：

- **抽象方法**: 需要在子类中实现具体的数据库操作逻辑
- **统一接口**: 为不同的数据库类型提供统一的查询接口
- **错误处理**: 子类实现应该包含适当的错误处理和状态报告
- **结果格式化**: 返回格式化的字符串结果，便于后续处理

**注意**: 此方法是数据库管理器的核心方法，所有具体的数据库操作都通过此方法执行。

"""
        pass

    def forward(self, statement: str) -> str:
        return self.execute_query(statement)

    @property
    def db_type(self) -> str:
        return self._db_type

    @property
    @abstractmethod
    def desc(self) -> str: pass

    @staticmethod
    def _is_dict_all_str(d):
        if not isinstance(d, dict):
            return False
        return all(isinstance(key, str) and (isinstance(value, str) or DBManager._is_dict_all_str(value))
                   for key, value in d.items())

    @staticmethod
    def _serialize_uncommon_type(obj):
        if not isinstance(obj, (int, str, float, bool, tuple, list, dict)):
            return str(obj)

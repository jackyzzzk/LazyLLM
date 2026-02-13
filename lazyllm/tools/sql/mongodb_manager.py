import json
from contextlib import contextmanager
from urllib.parse import quote_plus
import pydantic

from lazyllm.thirdparty import pymongo

from .db_manager import DBManager, DBResult, DBStatus


class CollectionDesc(pydantic.BaseModel):
    summary: str = ''
    schema_type: dict
    schema_desc: dict


class MongoDBManager(DBManager):
    """MongoDBManager是与MongoB数据库进行交互的专用工具。它提供了检查连接，获取数据库连接对象，执行查询的方法。

Args:
   user (str): MongoDB用户名
    password (str): MongoDB密码
    host (str): MongoDB服务器地址
    port (int): MongoDB服务器端口
    db_name (str): 数据库名称
    collection_name (str): 集合名称
    **kwargs: 额外配置参数，包括：
        - options_str (str): 连接选项字符串
        - collection_desc_dict (dict): 集合描述字典


Examples:
    >>> from lazyllm.components import MongoDBManager
    >>> mgr = MongoDBManager(
    ...     user="admin",
    ...     password="123456",
    ...     host="localhost",
    ...     port=27017,
    ...     db_name="mydb",
    ...     collection_name="books"
    ... )
    >>> result = mgr.execute_query('[{"$match": {"author": "Tolstoy"}}]')
    >>> print(result)
    ... '[{"title": "War and Peace", "author": "Tolstoy"}]'
    """
    MAX_TIMEOUT_MS = 5000

    def __init__(self, user: str, password: str, host: str, port: int, db_name: str, collection_name: str, **kwargs):
        super().__init__(db_type='mongodb')
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._db_name = db_name
        self._collection_name = collection_name
        self._collection = None
        self._options_str = kwargs.get('options_str')
        self._conn_url = self._gen_conn_url()
        self._collection_desc_dict = kwargs.get('collection_desc_dict')

    @property
    def db_name(self):
        return self._db_name

    @property
    def collection_name(self):
        return self._collection_name

    def _gen_conn_url(self) -> str:
        password = quote_plus(self._password)
        conn_url = (f'{self._db_type}://{self._user}:{password}@{self._host}:{self._port}/'
                    f'{("?" + self._options_str) if self._options_str else ""}')
        return conn_url

    @contextmanager
    def get_client(self):
        """这是一个上下文管理器，它创建并返回一个数据库会话连接对象，并在使用完成后自动关闭会话。
使用方式例如：

with mongodb_manager.get_client() as client:
    all_dbs = client.list_database_names()

**Returns:**

- pymongo.MongoClient: 连接 MongoDB 数据库的对象
"""
        client = pymongo.MongoClient(self._conn_url, serverSelectionTimeoutMS=self.MAX_TIMEOUT_MS)
        try:
            yield client
        finally:
            client.close()

    @property
    def desc(self):
        if self._desc is None:
            self.set_desc(schema_desc_dict=self._collection_desc_dict)
        return self._desc

    def set_desc(self, schema_desc_dict: dict):
        """对于MongoDBManager搭配LLM使用自然语言查询的文档集设置其必须的关键字描述。注意，查询需要用到的关系字都必须提供，因为MonoDB无法像SQL数据库一样获得表结构信息

Args:
    schema_desc_dict (dict): 文档集的关键字描述
"""
        self._collection_desc_dict = schema_desc_dict
        if schema_desc_dict is None:
            with self.get_client() as client:
                egs_one = client[self._db_name][self._collection_name].find_one()
                if egs_one is not None:
                    self._desc = 'Collection Example:\n'
                    self._desc += json.dumps(egs_one, ensure_ascii=False, indent=4)
        else:
            self._desc = ''
            try:
                collection_desc = CollectionDesc.model_validate(schema_desc_dict)
            except pydantic.ValidationError as e:
                raise ValueError(f'Validate input schema_desc_dict failed: {str(e)}')
            if not self._is_dict_all_str(collection_desc.schema_type):
                raise ValueError('schema_type shouble be str or nested str dict')
            if not self._is_dict_all_str(collection_desc.schema_desc):
                raise ValueError('schema_desc shouble be str or nested str dict')
            if collection_desc.summary:
                self._desc += f'Collection summary: {collection_desc.summary}\n'
            self._desc += 'Collection schema:\n'
            self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)
            self._desc += 'Collection schema description:\n'
            self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)

    def check_connection(self) -> DBResult:
        """检查当前MongoDBManager的连接状态。

**Returns:**

- DBResult: DBResult.status 连接成功(True), 连接失败(False)。DBResult.detail 包含失败信息
"""
        try:
            with pymongo.MongoClient(self._conn_url, serverSelectionTimeoutMS=self.MAX_TIMEOUT_MS) as client:
                _ = client.server_info()
            return DBResult()
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def execute_query(self, statement) -> str:
        str_result = ''
        try:
            pipeline_list = json.loads(statement)
            with self.get_client() as client:
                collection = client[self._db_name][self._collection_name]
                result = list(collection.aggregate(pipeline_list))
                str_result = json.dumps(result, ensure_ascii=False, default=self._serialize_uncommon_type)
        except Exception as e:
            str_result = f'MongoDB ERROR: {str(e)}'
        return str_result

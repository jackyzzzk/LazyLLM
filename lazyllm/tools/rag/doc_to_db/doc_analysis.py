from typing import TypedDict
from typing import Union, Tuple, List, get_type_hints
import lazyllm
from lazyllm import OnlineChatModule, TrainableModule
from ..data_loaders import DirectoryReader
from .prompts import PROMPTS as DOC_KWS_PROMPTS
import random
import re
import json
from lazyllm.thirdparty import tiktoken


class DocInfoSchemaItem(TypedDict):
    """文档信息结构中单个字段的定义。

Args:
    key (str): 字段名
    desc (str): 字段含义描述
    type (str): 字段的数据类型
"""
    key: str
    desc: str
    type: str


DocInfoSchema = List[DocInfoSchemaItem]


def validate_schema_item(given_dict: dict, typed_dict_schema: type[dict]) -> Tuple[bool, str]:
    type_hints = get_type_hints(typed_dict_schema)

    for key, value_type in type_hints.items():
        if key not in given_dict:
            return False, f'key {key} is missing'
        elif not isinstance(given_dict[key], value_type):
            return False, f'key {key} should be of type {value_type.__name__}'
    return True, 'Success'


def trim_content_by_token_num(tokenizer, doc_content: str, token_limit: int):
    current_doc_token_num = len(tokenizer.encode(doc_content))
    if current_doc_token_num > token_limit:
        ratio = token_limit / current_doc_token_num
        doc_content = doc_content[: int(len(doc_content) * ratio)]
    return doc_content


class DocGenreAnalyser:
    """用于分析文档所属的类别，例如合同、简历、发票等。通过读取文档内容，并结合大模型判断其类型。

Args:
    maximum_doc_num (int): 最多分析的文档数量，默认是 3。


Examples:
    >>> import lazyllm
    >>> from lazyllm.components.doc_info_extractor import DocGenreAnalyser
    >>> from lazyllm import OnlineChatModule
    >>> m = OnlineChatModule(source="openai")
    >>> analyser = DocGenreAnalyser()
    >>> genre = analyser.analyse_doc_genre(m, "path/to/document.txt")
    >>> print(genre)
    contract
    """
    ONE_DOC_TOKEN_LIMIT = 10000

    def __init__(self, maximum_doc_num=3):
        self._reader = DirectoryReader(None, {}, {})
        self._pattern = re.compile(r'```json(.+?)```', re.DOTALL)
        self._maximum_doc_num = maximum_doc_num
        self._tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        assert self._maximum_doc_num > 0

    def gen_detection_query(self, doc_path: str):
        """生成用于文档类型检测的查询。

Args:
    doc_path (str): 文档路径。

**Returns:**

- str: 返回格式化的查询字符串，包含文档内容和检测提示。

注意：
    生成的查询会自动根据 ONE_DOC_TOKEN_LIMIT 限制文档内容的长度。
"""
        root_nodes = self._reader.load_data([doc_path], None)
        doc_content = ''
        for root_node in root_nodes:
            doc_content += root_node.text + '\n'
        doc_content = trim_content_by_token_num(self._tiktoken_tokenizer, doc_content, self.ONE_DOC_TOKEN_LIMIT)
        query = DOC_KWS_PROMPTS['doc_type_detection'].format(doc_content=doc_content)
        query += '\nBelow is the content of each document sample.\n\n'
        return query

    def _extract_doc_type_from_response(self, str_response: str) -> str:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            try:
                res_dict = json.loads(extracted_content)
                if not isinstance(res_dict, dict) or 'doc_type' not in res_dict:
                    return ''
                return res_dict['doc_type']
            except Exception as e:
                lazyllm.LOG.warning(f'Exception: {str(e)}, response_str: {str_response}')
                return ''
        else:
            return ''

    def analyse_doc_genre(self, llm: Union[OnlineChatModule, TrainableModule], doc_path: str) -> str:
        """分析文档类型。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于分析的语言模型实例。
    doc_path (str): 要分析的文档路径。

**Returns:**

- str: 返回检测到的文档类型。如果检测失败则返回空字符串。
"""
        query = self.gen_detection_query(doc_path)
        response = llm(query)
        doc_genre = self._extract_doc_type_from_response(response)
        return doc_genre


class DocInfoSchemaAnalyser:
    """用于从文档中抽取出关键信息字段的结构，如字段名、描述、字段类型。可用于构建信息提取模板。

Args:
    maximum_doc_num (int): 用于生成schema的最大文档数量，默认是 3。


Examples:
    >>> from lazyllm.components.doc_info_extractor import DocInfoSchemaAnalyser
    >>> from lazyllm import OnlineChatModule
    >>> analyser = DocInfoSchemaAnalyser()
    >>> m = OnlineChatModule(source="openai")
    >>> schema = analyser.analyse_info_schema(m, "contract", ["doc1.txt", "doc2.txt"])
    >>> print(schema)
    [{'key': 'party_a', 'desc': 'The first party', 'type': 'str'}, ...]
    """
    ONE_DOC_TOKEN_LIMIT = 30000

    def __init__(self, maximum_doc_num=3):
        self._reader = DirectoryReader(None, {}, {})
        self._pattern = re.compile(r'```json(.+?)```', re.DOTALL)
        self._maximum_doc_num = maximum_doc_num
        self._tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        assert self._maximum_doc_num > 0

    def _gen_first_round_query(self, doc_type: str, doc_paths: list[str]):
        doc_contents = []
        for doc_path in doc_paths:
            root_nodes = self._reader.load_data([doc_path], None)
            doc_content = ''
            for root_node in root_nodes:
                doc_content += root_node.text + '\n'
            doc_content = trim_content_by_token_num(self._tiktoken_tokenizer, doc_content, self.ONE_DOC_TOKEN_LIMIT)
            doc_contents.append(doc_content)
        query = DOC_KWS_PROMPTS['kws_generation'].format(number=len(doc_contents), doc_type=doc_type)
        query += '\nBelow is the content of each document sample.\n\n'
        for i, doc_content in enumerate(doc_contents):
            query += f'Document {i + 1}:\n```\n{doc_content}\n```\n\n'
        return query

    def _extract_schema_from_response(self, str_response: str) -> List[dict]:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        empty_list = []
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            try:
                kws_list = json.loads(extracted_content)
                # in case of the list is in a dict, unpack it
                if isinstance(kws_list, dict):
                    values = list(kws_list.values())
                    if len(values) == 1 and isinstance(values[0], list):
                        return values[0]
                if not isinstance(kws_list, list):
                    lazyllm.LOG.warning(f'Excepted original type list but got {type(kws_list)} value: {kws_list}')
                    return empty_list
                return kws_list
            except Exception as e:
                lazyllm.LOG.warning(f'Exception: {str(e)}, response_str: {str_response}')
                return empty_list
        else:
            return empty_list

    def analyse_info_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], doc_type: str, doc_paths: list[str]
    ) -> DocInfoSchema:
        """分析文档信息模式的方法，用于从指定类型的文档中提取关键信息字段的结构定义。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于生成信息模式的LLM模型
    doc_type (str): 文档类型，用于指导LLM生成相应的信息模式
    doc_paths (list[str]): 文档路径列表，用于分析的信息来源

**Returns:**

- DocInfoSchema: 包含关键信息字段定义的模式列表，每个字段包含key、desc、type三个属性
"""
        RANDOM_SEED = 1331
        if len(doc_paths) > self._maximum_doc_num:
            doc_paths.sort()
            random.seed(RANDOM_SEED)
            doc_paths = random.sample(doc_paths, self._maximum_doc_num)
        first_round_query = self._gen_first_round_query(doc_type, doc_paths)
        first_response = llm(first_round_query)
        info_schema = self._extract_schema_from_response(first_response)
        for info_schema_item in info_schema:
            is_success, msg = validate_schema_item(info_schema_item, DocInfoSchemaItem)
            if not is_success:
                lazyllm.LOG.warning(f'Please Try Again! Invalid kws dict: {info_schema_item}, error_msg: {msg}')
                return []
        return info_schema


class DocInfoExtractor:
    """根据给定的字段结构（schema）从文档中抽取具体的关键信息值，返回格式为 key-value 字典。

Args:
    无


Examples:
    >>> from lazyllm.components.doc_info_extractor import DocInfoExtractor
    >>> from lazyllm import OnlineChatModule
    >>> extractor = DocInfoExtractor()
    >>> m = OnlineChatModule(source="openai")
    >>> schema = [{"key": "party_a", "desc": "Party A name", "type": "str"}]
    >>> info = extractor.extract_doc_info(m, "contract.txt", schema)
    >>> print(info)
    {'party_a': 'ABC Corp'}
    """
    ONE_DOC_TOKEN_LIMIT = 50000

    def __init__(self):
        self._reader = DirectoryReader(None, {}, {})
        self._pattern = re.compile(r'```json(.+?)```', re.DOTALL)
        self._tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    def _gen_extraction_query(self, doc_path: str, info_schema: DocInfoSchema, extra_desc: str) -> str:
        root_nodes = self._reader.load_data([doc_path], None)
        doc_content = ''
        for root_node in root_nodes:
            doc_content += root_node.text + '\n'
        doc_content = trim_content_by_token_num(self._tiktoken_tokenizer, doc_content, self.ONE_DOC_TOKEN_LIMIT)
        if not extra_desc:
            extra_desc = f'Extra description: \n{extra_desc}'
        query = DOC_KWS_PROMPTS['kws_extraction'].format(
            kws_desc=json.dumps(info_schema), extra_desc=extra_desc, doc_content=doc_content
        )
        return query

    def _extract_kws_value_from_response(self, str_response: str) -> dict:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        empty_dict = {}
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            try:
                kws_value = json.loads(extracted_content)
                if not isinstance(kws_value, dict):
                    lazyllm.LOG.warning(f'Excepted original type list but got {type(kws_value)}')
                    return empty_dict
                new_dict = {k: v for k, v in kws_value.items() if (isinstance(v, str) and v and v != 'None')}
                return new_dict
            except Exception as e:
                lazyllm.LOG.warning(f'Exception: {str(e)}, response_str: {str_response}')
                return empty_dict
        else:
            return empty_dict

    def _format_info_by_schema(self, info: dict, info_schema: DocInfoSchema):
        valid_keys = set([info_schema_item['key'] for info_schema_item in info_schema])
        return {k: v for k, v in info.items() if k in valid_keys}

    def extract_doc_info(
        self,
        llm: Union[OnlineChatModule, TrainableModule],
        doc_path: str,
        info_schema: DocInfoSchema,
        extra_desc: str = '',
    ) -> dict:
        """根据提供的字段结构（schema）从指定文档中抽取具体的关键信息值。

该方法使用大语言模型分析文档内容，根据预定义的字段结构提取相应的信息值，返回格式为 key-value 字典。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于文档信息抽取的大语言模型。
    doc_path (str): 要分析的文档路径。
    info_schema (DocInfoSchema): 字段结构定义，包含需要提取的字段信息。
    extra_desc (str, optional): 额外的描述信息，用于指导信息抽取。默认为空字符串。

**Returns:**

- dict: 提取出的关键信息字典，键为字段名，值为对应的信息值。
"""
        extraction_query = self._gen_extraction_query(doc_path, info_schema, extra_desc)
        response = llm(extraction_query)
        info: dict = self._extract_kws_value_from_response(response)
        info: dict = self._format_info_by_schema(info, info_schema)
        return info

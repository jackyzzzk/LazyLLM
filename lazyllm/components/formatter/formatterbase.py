import os
import json
import uuid
import hashlib
from typing import Callable, Optional, List, Union, Any

import lazyllm
from lazyllm.flow.flow import Pipeline

from ...common import LazyLLMRegisterMetaClass, package, Finalizer

def _is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        if s == 'None' or len(s) == 0:
            return False
        else:
            raise ValueError('Invalid number: ' + s + '. You can enter an integer, None or an empyt string.')

def _is_chat_message(msg):
    return isinstance(msg, dict) and 'role' in msg and 'content' in msg

class LazyLLMFormatterBase(metaclass=LazyLLMRegisterMetaClass):
    """此类是格式化器的基类，格式化器是模型输出结果的格式化器，用户可以自定义格式化器，也可以使用LazyLLM提供的格式化器。


Examples:
    >>> from lazyllm.components.formatter import LazyLLMFormatterBase
    >>> class MyFormatter(LazyLLMFormatterBase):
    ...     def __init__(self, formatter: str = None):
    ...         self._formatter = formatter
    ...         if self._formatter:
    ...             self._parse_formatter()
    ...         else:
    ...             self._slices = None
    ...     def _parse_formatter(self):
    ...         slice_str = self._formatter.strip()[1:-1]
    ...         slices = []
    ...         parts = slice_str.split(":")
    ...         start = int(parts[0]) if parts[0] else None
    ...         end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    ...         step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    ...         slices.append(slice(start, end, step))
    ...         self._slices = slices
    ...     def _load(self, data):
    ...         return [int(x) for x in data.strip('[]').split(',')]
    ...     def _parse_py_data_by_formatter(self, data):
    ...         if self._slices is not None:
    ...             result = []
    ...             for s in self._slices:
    ...                 if isinstance(s, slice):
    ...                     result.extend(data[s])
    ...                 else:
    ...                     result.append(data[int(s)])
    ...             return result
    ...         else:
    ...             return data
    ...
    >>> fmt = MyFormatter("[1:3]")
    >>> res = fmt.format("[1,2,3,4,5]")
    >>> print(res)
    [2, 3]
    """
    def _load(self, msg: str):
        return msg

    def _parse_py_data_by_formatter(self, py_data):
        raise NotImplementedError('This data parse function is not implemented.')

    def format(self, msg):
        """格式化输入消息。

Args:
    msg: 输入消息，可以是字符串或其他格式

**Returns:**

- 格式化后的数据，具体类型由子类实现决定
"""
        if _is_chat_message(msg): msg = msg['content']
        if isinstance(msg, str): msg = self._load(msg)
        return self._parse_py_data_by_formatter(msg)

    def __call__(self, *msg):
        return self.format(msg[0] if len(msg) == 1 else package(msg))

    def __or__(self, other):
        if not isinstance(other, LazyLLMFormatterBase):
            return NotImplemented
        return PipelineFormatter(other.__ror__(self))

    def __ror__(self, f: Callable) -> Pipeline:
        if isinstance(f, Pipeline):
            if not f._capture:
                _ = Finalizer(lambda: setattr(f, '_capture', True), lambda: setattr(f, '_capture', False))
            f._add(str(uuid.uuid4().hex) if len(f._item_names) else None, self)
            return f
        return Pipeline(f, self)


class PipelineFormatter(LazyLLMFormatterBase):
    """流水线格式化器，用于将数据处理流水线封装为格式化器。

该类将Pipeline实例包装为格式化器，支持通过管道操作符组合多个格式化器。

Args:
    formatter (Pipeline): 要封装的流水线实例
"""
    def __init__(self, formatter: Pipeline):
        self._formatter = formatter

    def _parse_py_data_by_formatter(self, py_data):
        return self._formatter(py_data)

    def __or__(self, other):
        if isinstance(other, LazyLLMFormatterBase):
            if isinstance(other, PipelineFormatter): other = other._formatter
            return PipelineFormatter(self._formatter | other)
        return NotImplemented


class JsonLikeFormatter(LazyLLMFormatterBase):
    """该类用于以类 JSON 的格式提取嵌套结构（如 dict、list、tuple）中的子字段内容。

其功能通过格式化字符串 `formatter` 来控制，格式类似于数组/字典的索引切片表达式。例如：

- `[0]` 表示取第 0 个元素
- `[0][{key}]` 表示取第 0 个元素并获取其中 key 字段
- `[0,1][{a,b}]` 表示同时提取第 0 和第 1 个对象的 a 和 b 字段
- `[::2]` 表示步长为 2 的切片
- `*[0][{x}]` 表示以包装格式返回处理后的数据（用于进一步结构化）

Args:
    formatter (str, optional): 控制提取规则的格式字符串。若为 None，则返回原始数据。


Examples:
    >>> from lazyllm.components.formatter.formatterbase import JsonLikeFormatter
    >>> formatter = JsonLikeFormatter("[{a,b}]")
    """
    class _ListIdxes(tuple): pass
    class _DictKeys(tuple): pass

    def __init__(self, formatter: Optional[str] = None):
        if formatter and formatter.startswith('*['):
            self._return_package = True
            self._formatter = formatter.strip('*')
        else:
            self._return_package = False
            self._formatter = formatter

        if self._formatter:
            assert '*' not in self._formatter, '`*` can only be used before `[` in the beginning'
            self._formatter = self._formatter.strip().replace('{', '[{').replace('}', '}]')
            self._parse_formatter()
        else:
            self._slices = None

    def _parse_formatter(self):
        # Remove the surrounding brackets
        assert self._formatter.startswith('[') and self._formatter.endswith(']')
        slice_str = self._formatter.strip()[1:-1]
        dimensions = slice_str.split('][')
        slices = []

        for dim in dimensions:
            if '{' in dim:
                slices.append(__class__._DictKeys(d.strip() for d in dim[1:-1].split(',') if d.strip()))
            elif ':' in dim:
                assert ',' not in dim, '[a, b:c] is not supported'
                parts = dim.split(':')
                start = int(parts[0]) if _is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and _is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and _is_number(parts[2]) else None
                slices.append(slice(start, end, step))
            elif ',' in dim:
                slices.append(__class__._ListIdxes(d.strip() for d in dim.split(',') if d.strip()))
            else:
                slices.append(dim.strip())
        self._slices = slices

    def _parse_py_data_by_formatter(self, data, *, slices=None):  # noqa C901
        def _impl(data, slice):
            if isinstance(data, (tuple, list)) and isinstance(slice, str):
                return data[int(slice)]
            if isinstance(slice, __class__._ListIdxes):
                if isinstance(data, dict): return [data[k] for k in slice]
                elif isinstance(data, (tuple, list)): return type(data)(data[int(k)] for k in slice)
                else: raise RuntimeError('Only tuple/list/dict is supported for [a,b,c]')
            if isinstance(slice, __class__._DictKeys):
                assert isinstance(data, dict)
                if len(slice) == 1 and slice[0] == ':': return data
                return {k: data[k] for k in slice}
            return data[slice]

        if slices is None: slices = self._slices
        if not slices: return data
        curr_slice = slices[0]
        if isinstance(curr_slice, slice):
            if isinstance(data, dict):
                assert curr_slice.start is None and curr_slice.stop is None and curr_slice.step is None, (
                    'Only {:} and [:] is supported in dict slice')
                curr_slice = __class__._ListIdxes(data.keys())
            elif isinstance(data, (tuple, list)):
                return type(data)(self._parse_py_data_by_formatter(d, slices=slices[1:])
                                  for d in _impl(data, curr_slice))
        if isinstance(curr_slice, __class__._DictKeys):
            return {k: self._parse_py_data_by_formatter(v, slices=slices[1:])
                    for k, v in _impl(data, curr_slice).items()}
        elif isinstance(curr_slice, __class__._ListIdxes):
            tp = package if self._return_package else list if isinstance(data, dict) else type(data)
            return tp(self._parse_py_data_by_formatter(r, slices=slices[1:]) for r in _impl(data, curr_slice))
        else: return self._parse_py_data_by_formatter(_impl(data, curr_slice), slices=slices[1:])


class PythonFormatter(JsonLikeFormatter):
    """预留格式化器类，用于支持 Python 风格的数据提取语法，待开发。

当前继承自 JsonLikeFormatter，无额外功能。
"""
    pass


class EmptyFormatter(LazyLLMFormatterBase):
    """此类是空的格式化器，即用户希望对模型的输出不做格式化，用户可以对模型指定该格式化器，也可以不指定(模型默认的格式化器就是空格式化器)


Examples:
    >>> import lazyllm
    >>> from lazyllm.components import EmptyFormatter
    >>> toc_prompt='''
    ... You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.
    ... 
    ... Please generate the corresponding list of nested dictionaries based on the following user input:
    ... 
    ... Example output:
    ... [
    ...     {
    ...         "title": "# Level 1 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
    ...     },
    ...     {
    ...         "title": "## Level 2 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
    ...     },
    ...     {
    ...         "title": "### Level 3 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
    ...     }
    ... ]
    ... User input is as follows:
    ... '''
    >>> query = "Please help me write an article about the application of artificial intelligence in the medical field."
    >>> m = lazyllm.TrainableModule("internlm2-chat-20b").prompt(toc_prompt).start()  # the model output without specifying a formatter
    >>> ret = m(query, max_new_tokens=2048)
    >>> print(f"ret: {ret!r}")
    'Based on your user input, here is the corresponding list of nested dictionaries:
    [
        {
            "title": "# Application of Artificial Intelligence in the Medical Field",
            "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."
        },
        {
            "title": "## AI in Medical Diagnosis",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."
        },
        {
            "title": "### AI in Medical Imaging",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."
        },
        {
            "title": "### AI in Drug Discovery and Development",
            "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."
        },
        {
            "title": "## AI in Medical Research",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."
        },
        {
            "title": "### AI in Genomics and Precision Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."
        },
        {
            "title": "### AI in Epidemiology and Public Health",
            "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."
        },
        {
            "title": "### AI in Clinical Trials",
            "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."
        },
        {
            "title": "## AI in Medical Practice",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."
        },
        {
            "title": "### AI in Patient Monitoring",
            "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."
        },
        {
            "title": "### AI in Personalized Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."
        },
        {
            "title": "### AI in Telemedicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."
        },
        {
            "title": "## AI in Medical Ethics and Policy",
            "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."
        }
    ]'
    >>> m = lazyllm.TrainableModule("internlm2-chat-20b").formatter(EmptyFormatter()).prompt(toc_prompt).start()  # the model output of the specified formatter
    >>> ret = m(query, max_new_tokens=2048)
    >>> print(f"ret: {ret!r}")
    'Based on your user input, here is the corresponding list of nested dictionaries:
    [
        {
            "title": "# Application of Artificial Intelligence in the Medical Field",
            "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."
        },
        {
            "title": "## AI in Medical Diagnosis",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."
        },
        {
            "title": "### AI in Medical Imaging",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."
        },
        {
            "title": "### AI in Drug Discovery and Development",
            "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."
        },
        {
            "title": "## AI in Medical Research",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."
        },
        {
            "title": "### AI in Genomics and Precision Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."
        },
        {
            "title": "### AI in Epidemiology and Public Health",
            "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."
        },
        {
            "title": "### AI in Clinical Trials",
            "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."
        },
        {
            "title": "## AI in Medical Practice",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."
        },
        {
            "title": "### AI in Patient Monitoring",
            "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."
        },
        {
            "title": "### AI in Personalized Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."
        },
        {
            "title": "### AI in Telemedicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."
        },
        {
            "title": "## AI in Medical Ethics and Policy",
            "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."
        }
    ]'
    """
    def _parse_py_data_by_formatter(self, msg: str):
        return msg

class FunctionCallFormatter(LazyLLMFormatterBase):
    """函数调用格式化器，用于处理包含函数调用信息的消息字典。

该格式化器专门用于处理函数调用场景下的模型输出，只提取字典中的 'role'、'content' 和 'tool_calls' 字段，过滤掉其他不需要的字段。

主要用于 FunctionCall 等工具调用相关的功能模块。

Args:
    无参数，直接实例化使用。

注意:
    - 输入必须是字典类型，否则会抛出断言错误
    - 只保留字典中存在的 'role'、'content'、'tool_calls' 字段


Examples:
    >>> from lazyllm.components.formatter.formatterbase import FunctionCallFormatter
    >>> formatter = FunctionCallFormatter()
    >>> 
    >>> # 处理包含函数调用的消息
    >>> msg = {
    ...     'role': 'assistant',
    ...     'content': 'I will call a function to get the weather.',
    ...     'tool_calls': [
    ...         {
    ...             'id': 'call_123',
    ...             'type': 'function',
    ...             'function': {
    ...                 'name': 'get_weather',
    ...                 'arguments': '{"location": "Beijing"}'
    ...             }
    ...         }
    ...     ],
    ...     'other_field': 'will be filtered'
    ... }
    >>> result = formatter.format(msg)
    >>> print(result)
    {'role': 'assistant', 'content': 'I will call a function to get the weather.', 'tool_calls': [{'id': 'call_123', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location": "Beijing"}'}}]}
    >>> 
    >>> # 处理只包含部分字段的消息
    >>> msg2 = {
    ...     'role': 'assistant',
    ...     'content': 'Hello, how can I help you?'
    ... }
    >>> result2 = formatter.format(msg2)
    >>> print(result2)
    {'role': 'assistant', 'content': 'Hello, how can I help you?'}
    """
    def format(self, msg):
        assert isinstance(msg, dict), 'FunctionCallFormatter only supports dict input.'
        return {k: msg[k] for k in ('role', 'content', 'tool_calls') if k in msg}

LAZYLLM_QUERY_PREFIX = '<lazyllm-query>'

def encode_query_with_filepaths(query: str = None, files: Union[str, List[str]] = None) -> str:
    """将查询文本和文件路径编码为带有文档上下文的结构化字符串格式。

当指定文件路径时，该函数会将查询内容与文件路径打包成 JSON 格式，并在前缀 ``__lazyllm_docs__`` 的基础上编码返回。否则仅返回原始查询文本。

Args:
    query (str): 用户查询字符串，默认为空字符串。
    files (str or List[str]): 与查询相关的文档路径，可为单个字符串或字符串列表。

**Returns:**

- str: 编码后的结构化查询字符串，或原始查询。

Raises:
    AssertionError: 如果 `files` 不是字符串或字符串列表，或列表中元素类型错误。


Examples:
    >>> from lazyllm.components.formatter import encode_query_with_filepaths

    >>> # Encode a query along with associated documentation files
    >>> encode_query_with_filepaths("Generate questions based on the document", files=["a.md"])
    '<lazyllm-query>{"query": "Generate questions based on the document", "files": ["a.md"]}'
    """
    query = query if query else ''
    if files:
        if isinstance(files, str): files = [files]
        assert isinstance(files, list), 'files must be a list.'
        assert all(isinstance(item, str) for item in files), 'All items in files must be strings'
        return LAZYLLM_QUERY_PREFIX + json.dumps({'query': query, 'files': files})
    else:
        return query

def decode_query_with_filepaths(query_files: str) -> Union[dict, str]:
    """将结构化查询字符串解析为包含原始查询和文件路径的字典格式。

当输入字符串以特殊前缀 ``__lazyllm_docs__`` 开头时，函数会尝试从中提取 JSON 格式的查询信息；否则将原样返回字符串内容。

Args:
    query_files (str): 编码后的查询字符串，可能包含文档路径和查询内容。

**Returns:**

- Union[dict, str]: 若为结构化格式则返回包含 'query' 和 'files' 的字典，否则返回原始查询字符串。

Raises:
    AssertionError: 如果输入参数不是字符串类型。
    ValueError: 如果字符串为结构化格式但解析 JSON 失败。


Examples:
    >>> from lazyllm.components.formatter import decode_query_with_filepaths

    >>> # Decode a structured query with files
    >>> decode_query_with_filepaths('<lazyllm-query>{"query": "Summarize the content", "files": ["doc.md"]}')
    {'query': 'Summarize the content', 'files': ['doc.md']}

    >>> # Decode a plain string without files
    >>> decode_query_with_filepaths("This is just a simple question")
    'This is just a simple question'
    """
    assert isinstance(query_files, str), 'query_files must be a str.'
    query_files = query_files.strip()
    if query_files.startswith(LAZYLLM_QUERY_PREFIX):
        try:
            obj = json.loads(query_files[len(LAZYLLM_QUERY_PREFIX):])
            return obj
        except json.JSONDecodeError as e:
            raise ValueError(f'JSON parsing failed: {e}')
    else:
        return query_files

def lazyllm_merge_query(*args: str) -> str:
    """将多个查询字符串（可能包含文档路径）合并为一个统一的结构化查询字符串。

每个输入参数可以是普通查询字符串或由 ``encode_query_with_filepaths`` 编码后的结构化字符串。函数会自动解码、拼接查询文本，并合并所有涉及的文档路径，最终重新编码为统一的查询格式。

Args:
    *args (str): 多个查询字符串。每个字符串可以是普通文本或已编码的带文件路径的结构化查询。

**Returns:**

- str: 合并后的结构化查询字符串，包含统一的查询内容与文件路径。


Examples:
    >>> from lazyllm.components.formatter import encode_query_with_filepaths, lazyllm_merge_query

    >>> # Merge two structured queries with English content and associated files
    >>> q1 = encode_query_with_filepaths("Please summarize document one", files=["doc1.md"])
    >>> q2 = encode_query_with_filepaths("Add details from document two", files=["doc2.md"])
    >>> lazyllm_merge_query(q1, q2)
    '<lazyllm-query>{"query": "Please summarize document oneAdd details from document two", "files": ["doc1.md", "doc2.md"]}'

    >>> # Merge plain English text queries without documents
    >>> lazyllm_merge_query("What is AI?", "Explain deep learning.")
    'What is AI?Explain deep learning.'
    """
    if len(args) == 1:
        return args[0]
    for item in args:
        assert isinstance(item, str), 'Merge object must be str!'
    querys = ''
    files = []
    for item in args:
        decode = decode_query_with_filepaths(item)
        if isinstance(decode, dict):
            querys += decode['query']
            files.extend(decode['files'])
        else:
            querys += decode
    return encode_query_with_filepaths(querys, files)

def _lazyllm_get_file_list(files: Any) -> list:
    if isinstance(files, str):
        decode = decode_query_with_filepaths(files)
        if isinstance(decode, str):
            return [decode]
        if isinstance(decode, dict):
            return decode['files']
    elif isinstance(files, dict) and set(files.keys()) == {'query', 'files'}:
        return files['files']
    elif isinstance(files, list) and all(isinstance(item, str) for item in files):
        return files
    else:
        raise TypeError(f'Not supported type: {type(files)}.')

class FileFormatter(LazyLLMFormatterBase):
    """用于处理带文档上下文的查询字符串格式转换的格式化器。

支持三种模式：

- "decode"：将结构化查询字符串解码为包含 query 和 files 的字典。
- "encode"：将包含 query 和 files 的字典编码为结构化查询字符串。
- "merge"：将多个结构化查询字符串合并为一个整体查询。

Args:
    formatter (str): 指定操作模式，可为 "decode"、"encode" 或 "merge"（默认为 "decode"）。


Examples:
    >>> from lazyllm.components.formatter import FileFormatter

    >>> # Decode mode
    >>> fmt = FileFormatter('decode')
    """

    def __init__(self, formatter: str = 'decode'):
        self._mode = formatter.strip().lower()
        assert self._mode in ('decode', 'encode', 'merge')

    def _parse_py_data_by_formatter(self, py_data):
        if self._mode == 'merge':
            if isinstance(py_data, str):
                return py_data
            assert isinstance(py_data, package)
            return lazyllm_merge_query(*py_data)

        if isinstance(py_data, package):
            res = []
            for i_data in py_data:
                res.append(self._parse_py_data_by_formatter(i_data))
            return package(res)
        elif isinstance(py_data, (str, dict)):
            return self._decode_one_data(py_data)
        else:
            return py_data

    def _decode_one_data(self, py_data):
        if self._mode == 'decode':
            if isinstance(py_data, str):
                return decode_query_with_filepaths(py_data)
            else:
                return py_data
        else:
            if isinstance(py_data, dict) and 'query' in py_data and 'files' in py_data:
                return encode_query_with_filepaths(**py_data)
            else:
                return py_data


def file_content_hash(value):
    res = decode_query_with_filepaths(value)
    if isinstance(res, str):
        return hashlib.md5(value.encode()).hexdigest()
    query = res['query']
    file_path_list = res['files']

    hash_obj = hashlib.md5()
    hash_obj.update(query.encode('utf-8'))

    search_paths = [
        '',
        lazyllm.config['temp_dir'],
    ]
    for file_path in file_path_list:
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            full_path = None
            for base_path in search_paths:
                candidate_path = os.path.join(base_path, file_path) if base_path else file_path
                if os.path.exists(candidate_path) and os.path.isfile(candidate_path):
                    full_path = candidate_path
                    break
            if full_path is None:
                full_path = file_path
        try:
            with open(full_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
        except OSError:
            lazyllm.LOG.debug(f'Error: File not found or cannot be read: {full_path}')
            hash_obj.update(full_path.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16)


def proccess_path_recursively(value, process_func):
    if isinstance(value, str):
        if LAZYLLM_QUERY_PREFIX in value:
            res = decode_query_with_filepaths(value)
            if res['files']:
                replace_path = []
                for file_path in res['files']:
                    replace_path.append(process_func(file_path))
                res['files'] = replace_path
            return encode_query_with_filepaths(res['query'], res['files'])
        else:
            return value
    elif isinstance(value, (list, tuple)):
        process_items = []
        for item in value:
            process_items.append(proccess_path_recursively(item, process_func))
        if isinstance(value, tuple):
            return tuple(process_items)
        return process_items
    elif isinstance(value, dict):
        process_dict = {}
        for key, val in value.items():
            process_dict[key] = proccess_path_recursively(val, process_func)
        return process_dict
    elif isinstance(value, set):
        process_set = set()
        for item in value:
            process_set.add(proccess_path_recursively(item, process_func))
        return process_set
    else:
        return value

def path_relative_to_absolute(path):
    if os.path.isabs(path):
        return path
    absolute_path = os.path.join(lazyllm.config['temp_dir'], path)
    if os.path.exists(absolute_path):
        return os.path.abspath(absolute_path)
    else:
        return path

def path_absolute_to_relative(path):
    if not os.path.isabs(path):
        return path
    temp_dir_abs = os.path.abspath(lazyllm.config['temp_dir'])
    if path.startswith(temp_dir_abs):
        relative_path = path[len(temp_dir_abs):]
        if relative_path.startswith(os.sep):
            relative_path = relative_path[1:]
        return relative_path
    else:
        return path

def transform_path(value, mode='r2a'):
    assert mode in ('r2a', 'a2r')
    if mode == 'r2a':
        return proccess_path_recursively(value, path_relative_to_absolute)
    else:
        return proccess_path_recursively(value, path_absolute_to_relative)

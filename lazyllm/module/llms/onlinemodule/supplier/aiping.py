import requests
import lazyllm
from typing import Tuple, List, Dict, Union
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineRerankModuleBase, LazyLLMOnlineText2ImageModuleBase
)
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase

TIMEOUT = 300

class AipingChat(OnlineChatModuleBase, FileHandlerBase):
    """AipingChat 是 AIPing 的在线聊天模块，继承自 OnlineChatModuleBase 和 FileHandlerBase。

提供与 AIPing 平台大语言模型交互的接口，支持对话生成、文件处理以及模型微调等功能。支持多种模型，包括视觉语言模型（VLM）如 Qwen2.5-VL、Qwen3-VL、GLM-4.5V、GLM-4.6V 等。

Args:
    base_url (str): API 基础 URL，默认为 "https://aiping.cn/api/v1/"。
    model (str): 使用的模型名称，默认为 "DeepSeek-R1"。
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (bool): 是否开启流式输出，默认为 True。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的参数。

功能特点:
    1. 支持多种大语言模型，包括通用对话模型和视觉语言模型
    2. 支持流式输出，提升用户体验
    3. 集成文件处理功能，支持微调数据格式验证和转换
    4. 内置系统提示："You are an intelligent assistant developed by AIPing. You are a helpful assistant."
    5. 支持 API Key 验证，确保服务安全性
"""
    VLM_MODEL_PREFIX = [
        'Qwen2.5-VL-',
        'Qwen3-VL-',
        'GLM-4.5V',
        'GLM-4.6V'
    ]

    def __init__(self, base_url: str = 'https://aiping.cn/api/v1/', model: str = 'DeepSeek-R1',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['aiping_api_key'], base_url=base_url, model_name=model,
                         stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by AIPing. You are a helpful assistant.'

    def _validate_api_key(self):
        try:
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=TIMEOUT)
            return response.status_code == 200
        except Exception:
            return False


class AipingEmbed(LazyLLMOnlineEmbedModuleBase):
    """ AIPing 文本嵌入模块，继承自 OnlineEmbeddingModuleBase。

提供与 AIPing 文本嵌入服务交互的接口，支持将文本转换为向量表示，支持批量处理。

Args:
    embed_url (str): 嵌入 API 的 URL，默认为 "https://aiping.cn/api/v1/embeddings"。
    embed_model_name (str): 使用的嵌入模型名称，默认为 "text-embedding-v1"。
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    batch_size (int): 批处理大小，默认为 16。
    **kw: 其他传递给基类的参数。

功能特点:
    1. 将文本转换为高维向量表示
    2. 支持批量文本处理，提高效率
    3. 可配置的批处理大小，适应不同性能需求
    4. 与 AIPing  API 无缝集成
"""
    def __init__(self, embed_url: str = 'https://aiping.cn/api/v1/embeddings',
                 embed_model_name: str = 'text-embedding-v1', api_key: str = None,
                 batch_size: int = 16, **kw):
        super().__init__(embed_url, api_key or lazyllm.config['aiping_api_key'],
                         embed_model_name, batch_size=batch_size, **kw)


class AipingRerank(LazyLLMOnlineRerankModuleBase):
    """ AIPing 重排序模块，继承自 OnlineEmbeddingModuleBase。

提供与 AIPing 重排序服务交互的接口，用于对文档列表根据查询相关性进行重新排序。该模块返回一个包含文档索引和相关性得分的元组列表。

Args:
    embed_url (str): 重排序 API 的 URL，默认为 "https://aiping.cn/api/v1/rerank"。
    embed_model_name (str): 使用的重排序模型名称，默认为 "Qwen3-Reranker-0.6B"。
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    **kw: 其他传递给基类的参数。

属性:
    type (str): 返回模型类型，固定为 "RERANK"。

功能特点:
    1. 根据查询对文档列表进行相关性重排序
    2. 支持自定义排序参数（top_n 等）
    3. 返回每个文档的索引和相关性得分
    4. 适用于搜索结果优化和文档推荐场景
"""
    def __init__(self, embed_url: str = 'https://aiping.cn/api/v1/rerank',
                 embed_model_name: str = 'Qwen3-Reranker-0.6B', api_key: str = None, **kw):
        super().__init__(embed_url, api_key or lazyllm.config['aiping_api_key'],
                         embed_model_name, **kw)

    @property
    def type(self):
        return 'RERANK'

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            'model': self._embed_model_name,
            'query': query,
            'documents': documents,
            'top_n': top_n
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        results = response.get('results', [])
        if not results:
            return []
        return [(result['index'], result['relevance_score']) for result in results]


class AipingText2Image(LazyLLMOnlineText2ImageModuleBase):
    """ AIPing 文本生成图像模块，继承自 OnlineMultiModalBase。

提供与 AIPing 图像生成服务交互的接口，支持根据文本描述生成图像。支持负面提示、图像数量、尺寸和随机种子等参数。

Args:
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    model_name (str): 使用的模型名称，默认为 "Qwen-Image"。
    base_url (str): API 基础 URL，默认为 "https://aiping.cn/api/v1/"。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给基类的参数。

功能特点:
    1. 根据文本提示生成高质量图像
    2. 支持负面提示，过滤不想要的图像特征
    3. 可配置生成图像的数量（n 参数）
    4. 支持多种图像尺寸规格
    5. 支持随机种子控制，确保结果可重现
    6. 自动下载生成的图像并编码为文件格式
    7. 默认负面提示："模糊，低质量"

注意:
    - 该模块会自动下载生成的图像到本地文件
    - 返回结果会包含文件路径信息，便于后续处理
"""
    def __init__(self, api_key: str = None, model_name: str = 'Qwen-Image',
                 base_url: str = 'https://aiping.cn/api/v1/',
                 return_trace: bool = False, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key or lazyllm.config['aiping_api_key'],
                         return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations'
        self._base_url = base_url

    def _make_request(self, endpoint, payload, timeout=TIMEOUT):
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self._base_url}{endpoint}'

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            lazyllm.LOG.error(f'Request failed: {e}')
            raise

    def _forward(self, input: str = None, negative_prompt: str = None, n: int = None,
                 size: str = None, seed: int = None, **kwargs):
        if not input:
            raise ValueError('Prompt is required')

        input_params = {
            'prompt': input,
            'negative_prompt': negative_prompt or '模糊，低质量'
        }

        extra_body = {}

        if n is not None:
            extra_body['n'] = n

        if size is not None:
            extra_body['size'] = size

        if seed is not None:
            extra_body['seed'] = seed

        payload = {
            'model': self._model_name,
            'input': input_params
        }

        if extra_body:
            payload['extra_body'] = extra_body

        try:
            result = self._make_request(self._endpoint, payload)

            images = result.get('data')
            if not images or not isinstance(images, list) or not images:
                raise ValueError(f'Unexpected response format: {result}')

            image_urls = [img.get('url') for img in images if img.get('url')]
            if not image_urls:
                raise ValueError(f'No image URLs found in response: {result}')

            return encode_query_with_filepaths(None, bytes_to_file([requests.get(url).content for url in image_urls]))

        except Exception as e:
            lazyllm.LOG.error(f'Failed to generate image: {e}')
            raise Exception(f'Failed to generate image: {str(e)}')

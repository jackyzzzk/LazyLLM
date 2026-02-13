import lazyllm
from typing import Dict, List, Union
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineMultimodalEmbedModuleBase, LazyLLMOnlineText2ImageModuleBase
)
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime
from lazyllm import LOG


class DoubaoChat(OnlineChatModuleBase):
    """豆包（Doubao）在线聊天模块，继承自 OnlineChatModuleBase。  
封装了对字节跳动 Doubao API 的调用，用于进行多轮问答交互。默认使用模型 `doubao-1-5-pro-32k-250115`，支持流式输出和调用链追踪。

Args:
    model (str): 使用的模型名称，默认为 `doubao-1-5-pro-32k-250115`。
    base_url (str): API 基础 URL，默认为 "https://ark.cn-beijing.volces.com/api/v3/"。
    api_key (Optional[str]): Doubao API Key，若未提供，则从 lazyllm.config['doubao_api_key'] 读取。
    stream (bool): 是否启用流式输出，默认为 True。
    return_trace (bool): 是否返回调用链追踪信息，默认为 False。
    **kwargs: 其他传递给基类 OnlineChatModuleBase 的参数。
"""
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'
    VLM_MODEL_PREFIX = ['doubao-seed-1-6-vision', 'doubao-1-5-ui-tars']

    def __init__(self, model: str = None, base_url: str = 'https://ark.cn-beijing.volces.com/api/v3/',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoChat.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Doubao, an AI assistant. Your task is to provide appropriate responses '
                'and support to user\'s questions and requests.')

    def _validate_api_key(self):
        try:
            # Doubao (Volcano Engine) validates API key using a minimal chat request
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1  # Only generate 1 token for validation
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class DoubaoEmbed(LazyLLMOnlineEmbedModuleBase):
    """豆包嵌入类，继承自 OnlineEmbeddingModuleBase，封装了调用豆包在线文本嵌入服务的功能。  
通过指定服务接口 URL、模型名称及 API Key，支持远程获取文本向量表示。

Args:
    embed_url (Optional[str]): 豆包文本嵌入服务的接口 URL，默认指向北京区域的服务地址。
    embed_model_name (Optional[str]): 使用的豆包嵌入模型名称，默认为 "doubao-embedding-text-240715"。
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
"""
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings',
                 embed_model_name: str = 'doubao-embedding-text-240715',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__(embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)


class DoubaoMultimodalEmbed(LazyLLMOnlineMultimodalEmbedModuleBase):
    """豆包多模态嵌入类，继承自 OnlineEmbeddingModuleBase，封装了调用豆包在线多模态（文本+图像）嵌入服务的功能。  
支持将文本和图像输入转换为统一的向量表示，通过指定服务接口 URL、模型名称及 API Key，实现远程获取多模态向量。

Args:
    embed_url (Optional[str]): 豆包多模态嵌入服务的接口 URL，默认指向北京区域的服务地址。
    embed_model_name (Optional[str]): 使用的豆包多模态嵌入模型名称，默认为 "doubao-embedding-vision-241215"。
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
"""
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal',
                 embed_model_name: str = None,
                 api_key: str = None):
        embed_model_name = (embed_model_name or lazyllm.config['doubao_multimodal_embed_model_name']
                            or 'doubao-embedding-vision-241215')
        super().__init__(embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{'text': input}]
        elif isinstance(input, list):
            # Validate input format, at most 1 text segment + 1 image
            if len(input) == 0:
                raise ValueError('Input list cannot be empty')
            if len(input) > 2:
                raise ValueError('Input list must contain at most 2 items (1 text and/or 1 image)')
        else:
            raise ValueError('Input must be either a string or a list of dictionaries')

        json_data = {
            'input': input,
            'model': self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[float]:
        # Doubao multimodal embedding returns a single fused embedding
        return response['data']['embedding']


class DoubaoMultiModal():
    """豆包多模态模块，继承自 OnlineMultiModalBase，封装了调用豆包多模态服务的能力。  
可通过指定 API Key、模型名称和服务基础 URL，远程调用豆包接口进行多模态数据处理和特征提取。

Args:
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
    model_name (Optional[str]): 使用的豆包多模态模型名称。
    base_url (str): 豆包服务的基础 URL，默认指向北京区域的服务地址。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineMultiModalBase 的参数。
"""
    def __init__(self, api_key: str = None, url: str = ''):
        api_key = api_key or lazyllm.config['doubao_api_key']
        self._client = volcenginesdkarkruntime.Ark(base_url=url, api_key=api_key)


class DoubaoText2Image(LazyLLMOnlineText2ImageModuleBase, DoubaoMultiModal):
    """字节跳动豆包文生图模块，支持纯文本生成图像和图像编辑模型。

基于字节跳动豆包多模态模型的文生图、图像编辑功能，继承自DoubaoMultiModal，
提供高质量的文本到图像生成能力。

Args:
    api_key (str, optional): 豆包API密钥，默认为None。
    model_name (str, optional): 模型名称，默认为"doubao-seedream-3-0-t2i-250415"。
    return_trace (bool, optional): 是否返回追踪信息，默认为False。
    **kwargs: 其他传递给父类的参数。
"""
    MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'
    IMAGE_EDITING_MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'

    def __init__(self, api_key: str = None, model: str = None, url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        resolved_model = model or lazyllm.config['doubao_text2image_model_name'] or DoubaoText2Image.MODEL_NAME
        super().__init__(model=resolved_model, api_key=api_key, return_trace=return_trace, url=url, **kwargs)
        DoubaoMultiModal.__init__(self, api_key=api_key, url=url)

    def _forward(self, input: str = None, files: List[str] = None, n: int = 1, size: str = '1024x1024', seed: int = -1,
                 guidance_scale: float = 2.5, watermark: bool = True, model: str = None, url: str = None, **kwargs):
        has_ref_image = files is not None and len(files) > 0
        if self._type == LLMType.IMAGE_EDITING and not has_ref_image:
            LOG.warning(
                f'Image editing is enabled for model {self._model_name}, but no image file was provided. '
                f'Please provide an image file via the "files" parameter.'
            )
        if self._type != LLMType.IMAGE_EDITING and has_ref_image:
            msg = str(f'Image file was provided, but image editing is not enabled for model {self._model_name}. Please '
                      f'use default image-editing model {self.IMAGE_EDITING_MODEL_NAME} or other image-editing model.')
            raise ValueError(msg)

        if has_ref_image:
            image_results = self._load_images(files)
            contents = [f'data:image/png;base64,{base64_str}' for base64_str, _ in image_results]
        api_params = {
            'model': model,
            'prompt': input,
            'size': size,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'watermark': watermark,
            **kwargs
        }
        if has_ref_image:
            api_params['image'] = contents
            if n > 1:
                api_params['sequential_image_generation'] = 'auto'
                max_images = min(n, 15)
                sigo = volcenginesdkarkruntime.types.images.SequentialImageGenerationOptions
                api_params['sequential_image_generation_options'] = sigo(max_images=max_images)
        imagesResponse = self._client.images.generate(**api_params)
        image_contents = [requests.get(result.url).content for result in imagesResponse.data]
        return encode_query_with_filepaths(None, bytes_to_file(image_contents))

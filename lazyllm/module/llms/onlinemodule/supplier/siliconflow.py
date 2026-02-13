import requests
import lazyllm
from typing import Tuple, List, Dict, Union
from urllib.parse import urljoin
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase, LazyLLMOnlineRerankModuleBase,
    LazyLLMOnlineText2ImageModuleBase, LazyLLMOnlineTTSModuleBase
)
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase
from lazyllm import LOG


class SiliconFlowChat(OnlineChatModuleBase, FileHandlerBase):
    """SiliconFlow 模块，继承自 OnlineChatModuleBase 和 FileHandlerBase。

提供基于 SiliconFlow 平台的大语言模型对话能力，支持多种模型（包括视觉语言模型），并具备文件处理功能。

Args:
    base_url (str, optional): API 基础地址，默认为 "https://api.siliconflow.cn/v1/"
    model (str, optional): 使用的模型名称，默认为 "Qwen/QwQ-32B"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['siliconflow_api_key'] 中读取
    stream (bool, optional): 是否启用流式输出，默认为 True
    return_trace (bool, optional): 是否返回追踪信息，默认为 False
    **kwargs: 其他模型参数
"""
    VLM_MODEL_PREFIX = ['Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Instruct', 'deepseek-ai/deepseek-vl2',
                        'Qwen/Qwen3-VL-30B-A3B-Thinking', 'THUDM/GLM-4.1V-9B-Thinking']

    def __init__(self, base_url: str = 'https://api.siliconflow.cn/v1/', model: str = 'Qwen/QwQ-32B',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['siliconflow_api_key'], base_url=base_url, model_name=model,
                         stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant provided by SiliconFlow. You are a helpful assistant.'

    def _validate_api_key(self):
        try:
            # SiliconFlow validates API key using a minimal chat request
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class SiliconFlowEmbed(LazyLLMOnlineEmbedModuleBase):
    """SiliconFlow 向量嵌入模块，继承自 OnlineEmbeddingModuleBase。

提供基于 SiliconFlow 平台的文本嵌入（Embedding）功能，支持将文本转换为向量表示。

Args:
    embed_url (str, optional): 嵌入 API 的 URL，默认为 "https://api.siliconflow.cn/v1/embeddings"
    embed_model_name (str, optional): 使用的嵌入模型名称，默认为 "BAAI/bge-large-zh-v1.5"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['siliconflow_api_key'] 中读取
    batch_size (int, optional): 批处理大小，默认为 16
    **kw: 其他嵌入模块参数
"""
    def __init__(self, embed_url: str = 'https://api.siliconflow.cn/v1/embeddings',
                 embed_model_name: str = 'BAAI/bge-large-zh-v1.5', api_key: str = None,
                 batch_size: int = 16, **kw):
        super().__init__(embed_url, api_key or lazyllm.config['siliconflow_api_key'],
                         embed_model_name, batch_size=batch_size, **kw)


class SiliconFlowRerank(LazyLLMOnlineRerankModuleBase):
    """SiliconFlow 重排序模块，继承自 OnlineEmbeddingModuleBase。

提供基于 SiliconFlow 平台的文本重排序（Reranking）功能，用于对文档列表根据查询相关性进行重新排序。

Args:
    embed_url (str, optional): 重排序 API 的 URL，默认为 "https://api.siliconflow.cn/v1/rerank"
    embed_model_name (str, optional): 使用的重排序模型名称，默认为 "BAAI/bge-reranker-v2-m3"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['siliconflow_api_key'] 中读取
    **kw: 其他重排序模块参数

Returns:
    List[Tuple]: 包含排序结果的列表，每个元素为包含 'index'、'relevance_score' 的元组。
"""
    def __init__(self, embed_url: str = 'https://api.siliconflow.cn/v1/rerank',
                 embed_model_name: str = 'BAAI/bge-reranker-v2-m3', api_key: str = None, **kw):
        super().__init__(embed_url, api_key or lazyllm.config['siliconflow_api_key'], embed_model_name, **kw)

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict:
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
        return [(result['index'], result['relevance_score']) for result in results]


class SiliconFlowText2Image(LazyLLMOnlineText2ImageModuleBase):
    """SiliconFlow文生图模块，继承自OnlineMultiModalBase。

提供基于SiliconFlow的文本生成图像功能，支持根据文本描述生成图像，支持纯文本生成图像和图像编辑。

Args:
    api_key (str, optional): API密钥，默认为配置中的siliconflow_api_key
    model_name (str, optional): 模型名称，默认为"Qwen/Qwen-Image"
    base_url (str, optional): API基础URL，默认为"https://api.siliconflow.cn/v1/"
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
"""
    MODEL_NAME = 'Qwen/Qwen-Image'
    IMAGE_EDITING_MODEL_NAME = 'Qwen/Qwen-Image-Edit-2509'

    def __init__(self, api_key: str = None, model: str = None,
                 url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['siliconflow_api_key'],
                         model=model or SiliconFlowText2Image.MODEL_NAME, url=url, return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations'

    def _make_request(self, endpoint, payload, base_url=None, timeout=180):
        url = f'{(base_url or self._base_url)}{endpoint}'
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, files: List[str] = None, size: str = '1024x1024', url: str = None,
                 model: str = None, **kwargs):
        has_ref_image = files is not None and len(files) > 0
        reference_image_data = None
        if self._type == LLMType.IMAGE_EDITING and not has_ref_image:
            raise ValueError(
                f'Image editing is enabled for model {self._model_name}, but no image file was provided. '
                f'Please provide an image file via the "files" parameter.'
            )
        if self._type != LLMType.IMAGE_EDITING and has_ref_image:
            raise ValueError(
                f'Image file was provided, but image editing is not enabled for model {self._model_name}. '
                f'Please use default image-editing model {self.IMAGE_EDITING_MODEL_NAME} or other image-editing model'
            )

        payload = {
            'model': model,
            'prompt': input,
            **kwargs
        }
        if has_ref_image:
            for i, file in enumerate(files):
                reference_image_base64, _ = self._load_images(file)[0]
                reference_image_data = f'data:image/png;base64,{reference_image_base64}'
                if i == 0:
                    payload['image'] = reference_image_data
                elif i > 0:
                    payload[f'image{i + 1}'] = reference_image_data
        result = self._make_request(self._endpoint, payload)
        image_urls = [item['url'] for item in result.get('data', [])]
        if not image_urls:
            raise Exception('No images returned from API')
        image_results = self._load_images(image_urls)
        image_bytes = [data for _, data in image_results]
        if not image_bytes:
            raise Exception('Failed to download any images')
        file_paths = bytes_to_file(image_bytes)
        return encode_query_with_filepaths(None, file_paths)


class SiliconFlowTTS(LazyLLMOnlineTTSModuleBase):
    """SiliconFlow文本转语音模块，继承自OnlineMultiModalBase。

提供基于SiliconFlow的文本转语音(TTS)功能，支持将文本转换为音频文件。

Args:
    api_key (str, optional): API密钥，默认为配置中的siliconflow_api_key
    model_name (str, optional): 模型名称，默认为"fnlp/MOSS-TTSD-v0.5"
    base_url (str, optional): API基础URL，默认为"https://api.siliconflow.cn/v1/"
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
"""
    MODEL_NAME = 'fnlp/MOSS-TTSD-v0.5'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['siliconflow_api_key'],
                         model_name=model_name or SiliconFlowTTS.MODEL_NAME,
                         return_trace=return_trace, base_url=base_url, **kwargs)
        self._endpoint = 'audio/speech'

    def _make_binary_request(self, endpoint, payload, base_url=None, timeout=180):
        url = f'{(base_url or self._base_url)}{endpoint}'
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, response_format: str = 'mp3',
                 sample_rate: int = 44100, speed: float = 1.0,
                 voice: str = None, references=None, out_path: str = None,
                 url: str = None, model: str = None, **kwargs):

        if not voice:
            active_model = model
            if active_model == 'fnlp/MOSS-TTSD-v0.5':
                voice = 'fnlp/MOSS-TTSD-v0.5:alex'
            elif active_model == 'FunAudioLLM/CosyVoice2-0.5B':
                voice = 'FunAudioLLM/CosyVoice2-0.5B:alex'
            else:
                raise ValueError(
                    f'Default voice is only supported for models "fnlp/MOSS-TTSD-v0.5" and '
                    f'"FunAudioLLM/CosyVoice2-0.5B". For model "{active_model}", '
                    f'please provide a valid voice parameter.')
        payload = {
            'model': model,
            'input': input,
            'response_format': response_format,
            'sample_rate': sample_rate,
            'speed': speed,
            'voice': voice
        }

        if references:
            payload['references'] = references

        payload.update(kwargs)
        audio_content = self._make_binary_request(self._endpoint, payload, base_url=url, timeout=180)
        file_path = bytes_to_file([audio_content])[0]

        if out_path:
            with open(file_path, 'rb') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
            file_path = out_path

        result = encode_query_with_filepaths(None, [file_path])

        return result

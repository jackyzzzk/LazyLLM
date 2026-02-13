import lazyllm
from urllib.parse import urljoin
import requests
from ..base import OnlineChatModuleBase


class KimiChat(OnlineChatModuleBase):
    """KimiChat 类，继承自 OnlineChatModuleBase，封装了调用 Moonshot AI 提供的 Kimi 聊天服务的能力。  
可通过指定 API Key、模型名称和服务 URL，支持中文和英文的安全问答交互，并支持图像输入的 base64 格式处理。

Args:
    base_url (str): Kimi 服务的基础 URL，默认为 "https://api.moonshot.cn/"。
    model (str): 使用的 Kimi 模型名称，默认为 "moonshot-v1-8k"。
    api_key (Optional[str]): 访问 Kimi 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (bool): 是否开启流式输出，默认为 True。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的参数。
"""

    def __init__(self, base_url: str = 'https://api.moonshot.cn/', model: str = 'moonshot-v1-8k',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):

        super().__init__(api_key=api_key or lazyllm.config['kimi_api_key'], base_url=base_url,
                         model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Kimi, an AI assistant provided by Moonshot AI. You are better at speaking '
                'Chinese and English. You will provide users with safe, helpful, and accurate answers. '
                'At the same time, you will reject all answers involving terrorism, racial discrimination, '
                'pornographic violence, etc. Moonshot AI is a proper noun and cannot be translated '
                'into other languages.')

    def _get_chat_url(self, url):
        if url.rstrip('/').endswith('v1/chat/completions'):
            return url
        return urljoin(url, 'v1/chat/completions')

    def _format_vl_chat_image_url(self, image_url, mime):
        assert not image_url.startswith('http'), 'Kimi vision model only supports base64 format'
        assert mime is not None, 'Kimi Module requires mime info.'
        image_url = f'data:{mime};base64,{image_url}'
        return [{'type': 'image_url', 'image_url': {'url': image_url}}]

    def _format_vl_chat_query(self, query: str):
        return query

    def _validate_api_key(self):
        try:
            models_url = urljoin(self._base_url, 'v1/models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

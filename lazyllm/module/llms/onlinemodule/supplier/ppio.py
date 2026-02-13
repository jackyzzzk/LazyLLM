from typing import Tuple
import requests
import lazyllm
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase


# PPIO (Paiou Cloud) online model module.
# PPIO provides OpenAI-compatible API interface, supporting both streaming and non-streaming responses.
class PPIOChat(OnlineChatModuleBase):
    """PPIO（派欧云）在线聊天模块，继承自 OnlineChatModuleBase。  
封装了对 PPIO (Paiou Cloud) API 的调用，用于进行多轮问答交互。默认使用模型 `deepseek/deepseek-v3.2`，支持流式输出和调用链追踪。PPIO 提供 OpenAI 兼容的 API 接口。

Args:
    model (str): 使用的模型名称，默认为 `deepseek/deepseek-v3.2`。
    base_url (str): API 基础 URL，默认为 "https://api.ppinfra.com/openai"。
    api_key (Optional[str]): PPIO API Key，若未提供，则从 lazyllm.config['ppio_api_key'] 读取。
    stream (bool): 是否启用流式输出，默认为 True。
    return_trace (bool): 是否返回调用链追踪信息，默认为 False。
    **kwargs: 其他传递给基类 OnlineChatModuleBase 的参数。


Examples:
    >>> import lazyllm
    >>> # Set environment variable: export LAZYLLM_PPIO_API_KEY=your_api_key
    >>> # Or create config file ~/.lazyllm/config.json: {"ppio_api_key": "your_api_key"}
    >>> chat = lazyllm.OnlineChatModule(source='ppio', model='deepseek/deepseek-v3.2')
    >>> response = chat('Hello, how are you?')
    >>> print(response)
    """
    TRAINABLE_MODEL_LIST = []
    NO_PROXY = False

    # Initialize PPIO module.
    # Args:
    #     base_url: API base URL, defaults to 'https://api.ppinfra.com/openai'
    #     model: Model name, defaults to 'deepseek/deepseek-v3.2'
    #     api_key: API key, if not provided, will be read from config
    #     stream: Whether to use streaming output, defaults to True
    #     return_trace: Whether to return execution trace, defaults to False
    #     skip_auth: Whether to skip authentication, defaults to False
    #     **kw: Other parameters
    def __init__(self, base_url: str = 'https://api.ppinfra.com/openai/',
                 model: str = 'deepseek/deepseek-v3.2',
                 api_key: str = None, stream: bool = True,
                 return_trace: bool = False, skip_auth: bool = False, **kw):
        super().__init__(api_key=api_key or lazyllm.config['ppio_api_key'], base_url=base_url,
                         model_name=model, stream=stream, return_trace=return_trace, skip_auth=skip_auth, **kw)

    # Return PPIO system prompt.
    def _get_system_prompt(self):
        return 'You are a helpful AI assistant.'

    # Validate API key by sending a minimal chat request.
    def _validate_api_key(self):
        try:
            data = {'model': self._model_name, 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1}
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    # Chat API URL - PPIO endpoint is /openai/chat/completions.
    def _get_chat_url(self, url):
        base = (url or '').rstrip('/')
        if base.endswith('/chat/completions'):
            return url
        if not base.endswith(('/openai', '/v1')):
            base = f'{base}/openai/'
        else:
            base = f'{base}/'
        return urljoin(base, 'chat/completions')

    # PPIO does not support deployment, return model name and running status.
    def _create_deployment(self) -> Tuple[str, str]:
        return (self._model_name, 'RUNNING')

    # PPIO does not support deployment query, return running status.
    def _query_deployment(self, deployment_id) -> str:
        return 'RUNNING'

    def __repr__(self):
        return lazyllm.make_repr('Module', 'PPIO', name=self._model_name, url=self._base_url,
                                 stream=bool(self._stream), return_trace=self._return_trace)

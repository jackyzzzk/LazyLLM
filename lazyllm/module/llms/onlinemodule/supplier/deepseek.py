from urllib.parse import urljoin
import requests
import lazyllm
from ..base import OnlineChatModuleBase


class DeepSeekChat(OnlineChatModuleBase):
    """DeepSeek大语言模型接口模块。

Args:
    base_url (str): API基础URL，默认为"https://api.deepseek.com"
    model (str): 模型名称，默认为"deepseek-chat"
    api_key (str): API密钥，如果为None则从配置中获取
    stream (bool): 启用流式输出，默认为True
    return_trace (bool): 返回追踪信息，默认为False
    **kwargs: 其他传递给基类的参数
"""
    def __init__(self, base_url: str = 'https://api.deepseek.com', model: str = 'deepseek-chat',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['deepseek_api_key'],
                         base_url=base_url, model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by China\'s DeepSeek. You are a helpful assistanti.'

    def _validate_api_key(self):
        try:
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

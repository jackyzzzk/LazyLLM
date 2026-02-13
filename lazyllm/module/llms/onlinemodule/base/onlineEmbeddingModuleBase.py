from typing import Dict, List, Union
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from lazyllm import LOG
from lazyllm.components.utils.downloader.model_downloader import LLMType
from .utils import LazyLLMOnlineBase
from lazyllm.components.utils.downloader import ModelManager


class OnlineEmbeddingModuleBase(LazyLLMOnlineBase):
    """OnlineEmbeddingModuleBase是管理开放平台的嵌入模型接口的基类，用于请求文本获取嵌入向量。不建议直接对该类进行直接实例化。需要特定平台类继承该类进行实例化。

如果你需要支持新的开放平台的嵌入模型的能力，请让你自定义的类继承自OnlineEmbeddingModuleBase：

1. 如果新平台的嵌入模型的请求和返回数据格式都和openai一样，可以不用做任何处理，只传url和模型即可
2. 如果新平台的嵌入模型的请求或者返回的数据格式和openai不一样，需要重写_encapsulated_data或_parse_response方法。
3. 配置新平台支持的api_key到全局变量，通过lazyllm.config.add(变量名，类型，默认值，环境变量名)进行添加

Args:
    embed_url (str): 嵌入API的URL地址。
    api_key (str): API访问密钥。
    embed_model_name (str): 嵌入模型名称。
    return_trace (bool, optional): 是否返回追踪信息，默认为False。


Examples:
    >>> import lazyllm
    >>> from lazyllm.module import OnlineEmbeddingModuleBase
    >>> class NewPlatformEmbeddingModule(OnlineEmbeddingModuleBase):
    ...     def __init__(self,
    ...                 embed_url: str = '<new platform embedding url>',
    ...                 embed_model_name: str = '<new platform embedding model name>'):
    ...         super().__init__(embed_url, lazyllm.config['new_platform_api_key'], embed_model_name)
    ...
    >>> class NewPlatformEmbeddingModule1(OnlineEmbeddingModuleBase):
    ...     def __init__(self,
    ...                 embed_url: str = '<new platform embedding url>',
    ...                 embed_model_name: str = '<new platform embedding model name>'):
    ...         super().__init__(embed_url, lazyllm.config['new_platform_api_key'], embed_model_name)
    ...
    ...     def _encapsulated_data(self, text:str, **kwargs):
    ...         pass
    ...         return json_data
    ...
    ...     def _parse_response(self, response: dict[str, any]):
    ...         pass
    ...         return embedding
    """
    NO_PROXY = True
    __lazyllm_registry_disable__ = True

    def __init__(self, embed_url: str, api_key: str, embed_model_name: str, skip_auth: bool = False,
                 return_trace: bool = False, batch_size: int = 32, num_worker: int = 1, timeout: int = 10):
        super().__init__(api_key=api_key, skip_auth=skip_auth, return_trace=return_trace)
        self._embed_url = embed_url
        self._embed_model_name = embed_model_name
        self._batch_size = batch_size
        self._num_worker = num_worker
        self._timeout = timeout

        if ModelManager.get_model_type(embed_model_name) == 'rerank':
            self._batch_size = 1
        else:
            self._batch_size = batch_size

        if hasattr(self, '_set_embed_url'): self._set_embed_url()

    @property
    def type(self):
        return 'EMBED'

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    def forward(self, input: Union[List, str], url: str = None, model: str = None, **kwargs
                ) -> Union[List[float], List[List[float]]]:
        runtime_url = url or kwargs.pop('base_url', kwargs.pop('embed_url', None)) or self._embed_url
        runtime_model = model or kwargs.pop('model_name', kwargs.pop('embed_model_name', None)) or self._embed_model_name

        if runtime_model is not None:
            kwargs['model'] = runtime_model
        data = self._encapsulated_data(input, **kwargs)
        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        if isinstance(data, list):
            return self.run_embed_batch(input, data, proxies, runtime_url, **kwargs)
        else:
            with requests.post(runtime_url, json=data, headers=self._header, proxies=proxies,
                               timeout=self._timeout) as r:
                if r.status_code == 200:
                    return self._parse_response(r.json(), input=input)
                else:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

    def _encapsulated_data(self, input: Union[List, str], **kwargs):
        if isinstance(input, str):
            json_data = {
                'input': [input],
                'model': self._embed_model_name
            }
            if len(kwargs) > 0:
                json_data.update(kwargs)
            return json_data
        else:
            text_batch = [input[i: i + self._batch_size] for i in range(0, len(input), self._batch_size)]
            json_data = [{'input': texts, 'model': self._embed_model_name} for texts in text_batch]
            if len(kwargs) > 0:
                for i in range(len(json_data)):
                    json_data[i].update(kwargs)
            return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> Union[List[List[float]], List[float]]:
        data = response.get('data', [])
        if not data:
            raise Exception('no data received')
        if isinstance(input, str):
            return data[0].get('embedding', [])
        else:
            return [res.get('embedding', []) for res in data]

    def run_embed_batch(self, input: List, data: List, proxies, url: str = None, **kwargs):
        """执行批量嵌入处理的内部方法。

此方法负责处理批量文本嵌入请求，支持单线程和多线程两种处理模式。
当遇到请求失败时，会自动调整批处理大小并重试，提供健壮的错误处理机制。

Args:
    input (List): 原始的输入文本列表
    data (List): 封装好的批量请求数据列表
    proxies: 代理设置，如果NO_PROXY为True则设置为None
    url (str, optional): 本次请求使用的完整接口地址，默认为初始传入的 embed_url
    **kwargs: 其他关键字参数

**Returns:**

- 嵌入向量列表的列表，每个子列表对应一个输入文本的嵌入向量
"""
        ret = [[] for _ in range(len(input))]
        flag = False
        url = url or self._embed_url
        if self._num_worker == 1:
            with requests.Session() as session:
                while not flag:
                    for i in range(len(data)):
                        r = session.post(url, json=data[i], headers=self._header,
                                         proxies=proxies, timeout=self._timeout)
                        if r.status_code == 200:
                            vec = self._parse_response(r.json(), input=input)
                            start = i * self._batch_size
                            ret[start: start + len(vec)] = vec
                            if i == len(data) - 1:
                                flag = True
                        else:
                            error_msg = '\n'.join([c.decode('utf-8') for c in r.iter_content(None)])
                            if self._batch_size == 1 or r.status_code in [401, 429]:
                                raise requests.RequestException(error_msg)
                            else:
                                msg = f'Online embedding:{self._embed_model_name} post failed, adjust batch_size: '
                                msg = msg + f' from {self._batch_size} to {max(self._batch_size // 2, 1)}'
                                LOG.warning(msg)
                                self._batch_size = max(self._batch_size // 2, 1)
                                data = self._encapsulated_data(input, **kwargs)
                                break
        else:
            with ThreadPoolExecutor(max_workers=self._num_worker) as executor:
                while not flag:
                    futures = [executor.submit(requests.post, url, json=t, headers=self._header,
                                               proxies=proxies, timeout=self._timeout) for t in data]
                    fut_to_index = {fut: idx for idx, fut in enumerate(futures)}
                    for fut in as_completed(futures):
                        r = fut.result()
                        i = fut_to_index.pop(fut)
                        if r.status_code == 200:
                            vec = self._parse_response(r.json(), input=input)
                            start = i * self._batch_size
                            ret[start: start + len(vec)] = vec
                            if len(fut_to_index) == 0:
                                flag = True
                        else:
                            wait(futures)
                            error_msg = '\n'.join([c.decode('utf-8') for c in r.iter_content(None)])
                            if self._batch_size == 1 or r.status_code in [401, 429]:
                                raise requests.RequestException(error_msg)
                            else:
                                msg = f'Online embedding:{self._embed_model_name} post failed, adjust batch_size: '
                                msg = msg + f' from {self._batch_size} to {max(self._batch_size // 2, 1)}'
                                LOG.warning(msg)
                                self._batch_size = max(self._batch_size // 2, 1)
                                data = self._encapsulated_data(input, **kwargs)
                                break
        return ret

class LazyLLMOnlineEmbedModuleBase(OnlineEmbeddingModuleBase):
    __lazyllm_registry_key__ = LLMType.EMBED

class LazyLLMOnlineMultimodalEmbedModuleBase(OnlineEmbeddingModuleBase):
    __lazyllm_registry_key__ = LLMType.MULTIMODAL_EMBED

class LazyLLMOnlineRerankModuleBase(OnlineEmbeddingModuleBase):
    __lazyllm_registry_key__ = LLMType.RERANK

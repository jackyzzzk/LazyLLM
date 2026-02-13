from ....module import ModuleBase
from lazyllm import config, LazyLLMRegisterMetaClass
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Optional, Union, List
import random


config.add('cache_online_module', bool, False, 'CACHE_ONLINE_MODULE',
           description='Whether to cache the online module result. Use for unit test.')


def select_source_with_default_key(available_models, explicit_source: Optional[str] = None, type: str = ''):
    if explicit_source:
        assert explicit_source in available_models, f'Unsupported source: {explicit_source}'
        key_name = f'{explicit_source}_api_key'
        default_key = config[key_name] if key_name in config.get_all_configs() else None
        return explicit_source, default_key
    if (default_source := config['default_source']):
        return default_source, config['default_key']

    for candidate in available_models.keys():
        candidate = candidate[:-len(type)]
        if config[f'{candidate}_api_key']:
            return candidate, None

    excepted = [f'{config.prefix}_{k[:-len(type)]}_api_key'.upper() for k in available_models.keys()]
    raise KeyError(f'No api_key is configured for any of the models {available_models.keys()}. '
                   f'You can set one of those environment: {excepted}')


def check_and_add_config(key, description):
    if key.lower() not in config.get_all_configs():
        config.add(key, str, '', f'{key.upper()}', description=description)


class LazyLLMOnlineBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    """LazyLLM 在线模块基类，继承自 ModuleBase，并使用 LazyLLMRegisterMetaClass， 为所有在线服务模块提供统一的基础功能。  
该类封装了在线模块的通用行为，包括缓存机制和调试追踪功能，是构建各种在线API服务模块的基础类。

功能特性:
    - 继承 ModuleBase 的所有基础功能，包括子模块管理、钩子注册等。
    - 支持在线模块缓存机制，可通过配置控制是否启用缓存。
    - 提供调试追踪功能，便于问题排查和性能分析。
    - 作为所有在线服务模块（如聊天、嵌入、多模态等）的公共基类。

Args:
    return_trace (bool): 是否将推理结果写入 trace 队列，用于调试和追踪。默认为 ``False``。

使用场景:
    1. 作为在线聊天模块（OnlineChatModuleBase）的基类。
    2. 作为在线嵌入模块（OnlineEmbeddingModuleBase）的基类。
    3. 作为在线多模态模块（OnlineMultiModalBase）的基类。
    4. 为自定义在线服务模块提供统一的基础功能。
"""
    _model_series = None

    def __init__(self, api_key: Optional[Union[str, List[str]]],
                 skip_auth: Optional[bool] = False, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        if not skip_auth and not api_key: raise ValueError('api_key is required')
        self.__api_keys = '' if skip_auth else api_key
        self.__headers = [self._get_header(key) for key in (api_key if isinstance(api_key, list) else [api_key])]
        if config['cache_online_module']:
            self.use_cache()

    @property
    def series(self):
        return self.__class__._model_series

    @property
    def _api_key(self):
        return random.choice(self.__api_keys) if isinstance(self.__api_keys, list) else self.__api_keys

    @staticmethod
    def _get_header(api_key: str) -> dict:
        return {'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + api_key} if api_key else {})}

    def _get_empty_header(self, api_key: Optional[str] = None) -> dict:
        api_key = api_key or self._api_key
        return {'Authorization': f'Bearer {api_key}'} if api_key else None

    @property
    def _header(self):
        return random.choice(self.__headers)

    @staticmethod
    def __lazyllm_after_registry_hook__(cls, group_name: str, name: str, isleaf: bool):

        allowed = set(list(LLMType))
        config_type_dict = {
            LLMType.CHAT: ('_model_name', 'The default model name for '),
            LLMType.EMBED: ('_model_name', 'The default embed model name for '),
            LLMType.RERANK: ('_model_name', 'The default rerank model name for '),
            LLMType.MULTIMODAL_EMBED: ('_multimodal_embed_model_name', 'The default multimodal embed model name for '),
            LLMType.STT: ('_stt_model_name', 'The default stt model name for '),
            LLMType.TTS: ('_tts_model_name', 'The default tts model name for '),
            LLMType.TEXT2IMAGE: ('_text2image_model_name', 'The default text2image model name for '),
        }

        check_and_add_config(key='default_source', description='The default model source for online modules.')
        check_and_add_config(key='default_key', description='The default API key for online modules.')

        if group_name == '':
            assert name == 'online'
        elif not isleaf:
            assert group_name == 'online', 'The group can only be "online" here.'
            assert name.lower() in allowed, f'Registry key {name} not in list {allowed}'
        else:
            subgroup = group_name.split('.')[-1]
            assert name.lower().endswith(subgroup), f'Class name {name} must follow \
                the schema of <SupplierType>, like <Qwen{subgroup.capitalize()}>'
            cls._model_series = supplier = name[:-len(subgroup)].lower()

            check_and_add_config(key=f'{supplier}_api_key',
                                 description=f'The API key for {supplier}')

            if subgroup in config_type_dict:
                key_suffix, description = config_type_dict[subgroup]
                check_and_add_config(key=f'{supplier}{key_suffix}',
                                     description=f'{description}{supplier}')

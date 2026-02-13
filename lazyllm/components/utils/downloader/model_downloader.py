import os
import time
import functools
import threading
from abc import ABC, abstractmethod
from enum import Enum, EnumMeta

import lazyllm
from .model_mapping import model_name_mapping, model_provider, model_groups
from .model_directory import infer_model_type

lazyllm.config.add('model_source', str, 'modelscope', 'MODEL_SOURCE', description='The default model source to use.')
lazyllm.config.add('model_cache_dir', str, os.path.join(os.path.expanduser(lazyllm.config['home']), 'model'),
                   'MODEL_CACHE_DIR', description='The default model cache directory to use(Read and Write).')
lazyllm.config.add('model_path', str, '', 'MODEL_PATH', description='The default model path to use(ReadOnly).')
lazyllm.config.add('model_source_token', str, '', 'MODEL_SOURCE_TOKEN',
                   description='The default token for configed model source(hf or ms) to use.')
lazyllm.config.add('data_path', str, '', 'DATA_PATH', description='The default data path to use.')


class _CaseInsensitiveEnumMeta(EnumMeta):
    def __getitem__(cls, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            if isinstance(name, str):
                lowered = name.casefold()
                for m in cls:
                    if m.name.casefold() == lowered:
                        return m
            raise


_EQUIVALENT = {'sd': 'text2image'}


class LLMType(str, Enum, metaclass=_CaseInsensitiveEnumMeta):
    """LLMType 枚举类

该枚举用于表示不同类型的大模型（如 LLM、VLM、TTS 等）。
特点：
- 成员值为字符串（继承自 str）。
- 支持大小写不敏感的构造与比较：
    - 构造时，既可以用成员名称，也可以用成员值，大小写不敏感。
    - 比较时，可以直接与字符串进行比较，大小写不敏感。
    - 成员名称索引（如 LLMType['xxx']）同样大小写不敏感。

可用类型：
    - LLM
    - VLM
    - SD
    - TTS
    - STT
    - EMBED
    - REANK
    - CROSS_MODAL_EMBED
    - OCR


Examples:
    >>> LLMType("llm")
    <LLMType.LLM: 'LLM'>

    >>> LLMType("llm") == LLMType.LLM
    True

    >>> LLMType("LLM") == LLMType.LLM
    True

    >>> LLMType.LLM == "llm"
    True

    >>> LLMType.LLM == "LLM"
    True

    >>> LLMType("CROSS_modal_embed")
    <LLMType.CROSS_MODAL_EMBED: 'CROSS_MODAL_EMBED'>
    """
    LLM = 'LLM'
    CHAT = 'CHAT'
    VLM = 'VLM'
    SD = 'SD'
    TEXT2IMAGE = 'TEXT2IMAGE'
    IMAGE_EDITING = 'IMAGE_EDITING'
    TTS = 'TTS'
    STT = 'STT'
    EMBED = 'EMBED'
    RERANK = 'RERANK'
    CROSS_MODAL_EMBED = 'CROSS_MODAL_EMBED'
    MULTIMODAL_EMBED = 'MULTIMODALEMBED'
    OCR = 'OCR'

    @classmethod
    def _normalize(cls, value: str) -> str:
        v = value.casefold()
        return _EQUIVALENT.get(v, v)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            v = cls._normalize(value)
            for m in cls:
                if cls._normalize(m.value) == v:
                    return m
            for m in cls:
                if cls._normalize(m.name) == v:
                    return m
        return None

    def __eq__(self, other):
        if isinstance(other, str):
            return self._normalize(self.value) == self._normalize(other)
        if isinstance(other, LLMType):
            return self._normalize(self.value) == self._normalize(other.value)
        return NotImplemented

    def __hash__(self):
        return hash(self._normalize(self.value))


class ModelManager():
    """ModelManager 是 LazyLLM 提供的模型管理与下载工具类，支持本地搜索和 Huggingface/Modelscope 下载。  

Args:
    model_source (Optional[str]): 模型下载源，仅支持 ``huggingface`` 或 ``modelscope``。
        未提供时使用 LAZYLLM_MODEL_SOURCE，若未设置则默认 ``modelscope``。
    token (Optional[str]): 下载私有模型的访问令牌。未提供时使用 LAZYLLM_MODEL_SOURCE_TOKEN。
    model_path (Optional[str]): 冒号分隔的本地绝对路径列表，用于下载前搜索模型。未提供时使用 LAZYLLM_MODEL_PATH。
    cache_dir (Optional[str]): 本地缓存目录，用于存放下载的模型。未提供时使用 LAZYLLM_MODEL_CACHE_DIR，默认 ``~/.lazyllm/model``。

Static Methods:
    get_model_type(model: str) -> str
        返回指定模型类型，如 ``llm``、``chat``，未识别返回 ``llm``。
    get_model_prompt_keys(model: str) -> dict
        返回模型的 prompt key 映射字典。
    validate_model_path(model_path: str) -> bool
        检查目录下是否存在有效模型文件（扩展名: ``.pt``, ``.bin``, ``.safetensors``）。

Instance Methods:
    download(model: Optional[str] = '', call_back: Optional[Callable] = None) -> str | bool
        下载指定模型。流程：
        1. 在 model_path 列出的本地目录搜索；
        2. 未找到则在 cache_dir 下搜索；
        3. 仍未找到则从 model_source 下载并存放 cache_dir。

        Args:
            model (Optional[str]): 目标模型名称，可使用简略名称或下载源完整名称。
            call_back (Optional[Callable]): 下载进度回调函数，可选。


Examples:
    >>> from lazyllm.components import ModelManager
    >>> downloader = ModelManager(model_source='modelscope')
    >>> downloader.download('chatglm3-6b')
    """
    def __init__(self, model_source, token=lazyllm.config['model_source_token'],
                 cache_dir=lazyllm.config['model_cache_dir'], model_path=lazyllm.config['model_path']):
        self.model_source = model_source or lazyllm.config['model_source']
        self.token = token or None
        self.cache_dir = cache_dir
        self.model_paths = model_path.split(':') if len(model_path) > 0 else []
        if self.model_source == 'huggingface':
            self.hub_downloader = _HuggingfaceDownloader(token=self.token)
        else:
            self.hub_downloader = _ModelscopeDownloader(token=self.token)
            if self.model_source != 'modelscope':
                lazyllm.LOG.error('Only support Huggingface and Modelscope currently. '
                                  f'Unsupported model source: {self.model_source}. Forcing use of Modelscope.')

    @staticmethod
    @functools.lru_cache
    def get_model_type(model) -> str:
        """根据模型名称获取模型类型（如 LLM、VLM 等）。

Args:
    model (str): 模型名称或路径，必须为非空字符串。

**Returns:**

- str: 模型类型，如果无法匹配则返回 ``llm``。
"""
        assert isinstance(model, str) and len(model) > 0, f'model name should be a non-empty string, get {model}'
        __class__._try_add_mapping(model)
        for name, info in model_name_mapping.items():
            if 'type' not in info: continue

            model_name_set = {name.casefold()}
            for source in info['source']:
                model_name_set.add(info['source'][source].split('/')[-1].casefold())

            if model.split(os.sep)[-1].casefold() in model_name_set:
                return info['type']
        return infer_model_type(model)

    @staticmethod
    @functools.lru_cache
    def _get_model_name(model) -> str:
        search_string = os.path.basename(model)
        __class__._try_add_mapping(search_string)
        for model_name, sources in model_name_mapping.items():
            if model_name.lower() == search_string.lower() or any(
                    os.path.basename(source_file).lower() == search_string.lower()
                    for source_file in sources['source'].values()):
                return model_name
        return ''

    @staticmethod
    @functools.lru_cache
    def get_model_prompt_keys(model) -> dict:
        """获取指定模型的 prompt key 映射字典，用于推理时构建输入。  

Args:
    model (str): 模型名称或路径。

**Returns:**

- dict: 模型对应的 prompt key 映射，如果不存在则返回空字典
"""
        model_name = __class__._get_model_name(model)
        __class__._try_add_mapping(model_name)
        if model_name and 'prompt_keys' in model_name_mapping[model_name.lower()]:
            return model_name_mapping[model_name.lower()]['prompt_keys']
        else:
            return dict()

    @staticmethod
    def validate_model_path(model_path):
        """检查指定路径下是否存在有效的模型文件（.pt, .bin, .safetensors）。  

Args:
    model_path (str): 模型目录路径。

**Returns:**

- bool: 如果目录中存在模型文件返回 True，否则返回 False
"""
        extensions = {'.pt', '.bin', '.safetensors'}
        for _, _, files in os.walk(model_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    return True
        return False

    @staticmethod
    def _try_add_mapping(model):
        model_base = os.path.basename(model)
        model = model_base.lower()
        if model in model_name_mapping.keys():
            return
        matched_model_prefix = next((key for key in model_provider if model.startswith(key)), None)
        if matched_model_prefix:
            matching_keys = [key for key in model_groups.keys() if key in model]
            if matching_keys:
                matched_groups = max(matching_keys, key=len)
                model_name_mapping[model] = {
                    'prompt_keys': model_groups[matched_groups]['prompt_keys'],
                    'source': {k: v + '/' + model_base for k, v in model_provider[matched_model_prefix].items()}
                }

    def download(self, model='', call_back=None):
        """下载指定名称的模型，如果本地已有则直接返回本地路径；  
支持 Huggingface 和 Modelscope 平台的自动下载，并会在缓存目录创建符号链接以便统一管理。  

Args:
    model (str, optional): 模型名称或路径，默认为空字符串，表示不下载。
    call_back (Optional[Callable], optional): 下载进度回调函数，接受当前下载状态等参数。  

**Returns:**

- str | bool: 模型在本地的完整路径，如果下载失败返回 False
"""
        assert isinstance(model, str), 'model name should be a string.'
        if len(model) == 0 or model[0] in (os.sep, '.', '~') or os.path.isabs(model): return model
        if (model_at_path := self._model_exists_at_path(model)): return model_at_path
        if self.model_source == '' or self.model_source not in ('huggingface', 'modelscope'):
            lazyllm.LOG.error('model automatic downloads only support Huggingface and Modelscope currently.')
            return model

        self._try_add_mapping(model)
        if model_name_mapping.get(model.lower(), {}).get('download_by_other'): return model

        if model.lower() in model_name_mapping.keys() and \
                self.model_source in model_name_mapping[model.lower()]['source'].keys():
            full_model_dir = os.path.join(self.cache_dir, model)

            mapped_model_name = model_name_mapping[model.lower()]['source'][self.model_source]
            model_save_dir = self._do_download(mapped_model_name, call_back)
            if model_save_dir:
                # The code safely creates a symbolic link by removing any existing target.
                if os.path.exists(full_model_dir):
                    os.remove(full_model_dir)
                if os.path.islink(full_model_dir):
                    os.unlink(full_model_dir)
                os.symlink(model_save_dir, full_model_dir, target_is_directory=True)
                return full_model_dir
            return model_save_dir  # return False
        else:
            model_name_for_download = model

            if '/' not in model_name_for_download:
                # Try to figure out a possible model provider
                matched_model_prefix = next((key for key in model_provider if model.lower().startswith(key)), None)
                if matched_model_prefix and self.model_source in model_provider[matched_model_prefix]:
                    model_name_for_download = model_provider[matched_model_prefix][self.model_source] + '/' + model

            model_save_dir = self._do_download(model_name_for_download, call_back)
            return model_save_dir

    def _validate_token(self):
        return self.hub_downloader.verify_hub_token()

    def _validate_model_id(self, model_id):
        return self.hub_downloader._verify_model_id(model_id)

    def _model_exists_at_path(self, model_name):
        if len(self.model_paths) == 0:
            return None
        model_dirs = []

        # For short model name, get all possible names from the mapping.
        if model_name.lower() in model_name_mapping.keys():
            for source in ('huggingface', 'modelscope'):
                if source in model_name_mapping[model_name.lower()]['source'].keys():
                    model_dirs.append(model_name_mapping[model_name.lower()]['source'][source].replace('/', os.sep))
        model_dirs.append(model_name.replace('/', os.sep))

        for model_path in self.model_paths:
            if len(model_path) == 0: continue
            if model_path[0] != os.sep:
                lazyllm.LOG.warning(f'skipping path {model_path} as only absolute paths is accepted.')
                continue
            for model_dir in model_dirs:
                full_model_dir = os.path.join(model_path, model_dir)
                if self._is_model_valid(full_model_dir):
                    return full_model_dir
        return None

    def _is_model_valid(self, model_dir):
        if not os.path.isdir(model_dir):
            return False
        return any((True for _ in os.scandir(model_dir)))

    def _do_download(self, model='', call_back=None):
        model_dir = model.replace('/', os.sep)
        full_model_dir = os.path.join(self.cache_dir, self.model_source, model_dir)

        try:
            return self.hub_downloader.download(model, full_model_dir, call_back)
        # Use `BaseException` to capture `KeyboardInterrupt` and normal `Exceptioin`.
        except BaseException as e:  # noqa B036
            lazyllm.LOG.warning(f'Download encountered an error: {e}')
            if '401' in str(e) or 'Client Error' in str(e):
                raise RuntimeError('Authentication failed (401 Error). Please check your access token and '
                                   'permissions.  And set the token with the environment variable '
                                   'LAZYLLM_MODEL_SOURCE_TOKEN.')
            if not self.token and 'Permission denied' not in str(e):
                lazyllm.LOG.warning('Token is empty, which may prevent private models from being downloaded, '
                                    'as indicated by "the model does not exist." Please set the token with the '
                                    'environment variable LAZYLLM_MODEL_SOURCE_TOKEN to download private models.')
            raise RuntimeError(f'Model download failed for model: {model}, with error: {e}')
        return False

class _HubDownloader(ABC):

    def __init__(self, token=None):
        self._token = token

    @lazyllm.once_wrapper
    def _lazy_init(self):
        self._token = self._token if self._token and self._verify_hub_token(self._token) else None
        self._api = self._build_hub_api(self._token)

    @abstractmethod
    def _verify_hub_token(self, token): pass

    @abstractmethod
    def _build_hub_api(self, token): pass

    def _verify_model_id(self, model_id):
        self._lazy_init()
        return self._verify_model_id_impl(model_id)

    @abstractmethod
    def _verify_model_id_impl(self, model_id): pass

    def _do_download(self, model_id, model_dir):
        if not self._verify_model_id(model_id):
            lazyllm.LOG.warning(f'Invalid model id:{model_id}')
            return False
        return self._do_download_impl(model_id, model_dir)

    @abstractmethod
    def _do_download_impl(self, model_id, model_dir): pass

    def _get_repo_files(self, model_id):
        self._lazy_init()
        return self._get_repo_files_impl(model_id)

    @abstractmethod
    def _get_repo_files_impl(self, model_id): pass

    def _polling_progress(self, model_dir, total, polling_event, call_back):
        while not polling_event.is_set():
            n = self._get_current_files_size(model_dir)
            n = min(n, total)
            if callable(call_back):
                try:
                    call_back(n, total)
                except Exception as e:
                    lazyllm.LOG.error(f'Error in callback: {e}')
            time.sleep(1)

    def _get_current_files_size(self, model_dir):
        total_size = 0
        for dirpath, _, filenames in os.walk(model_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def _get_files_total_size(self, hub_model_info):
        size = 0
        for item in hub_model_info:
            size += item['Size']
        return size

    def download(self, model_id, model_dir, call_back=None):
        total = self._get_files_total_size(self._get_repo_files(model_id))
        if call_back:
            polling_event = threading.Event()
            polling_thread = threading.Thread(target=self._polling_progress,
                                              args=(model_dir, total, polling_event, call_back))
            polling_thread.daemon = True
            polling_thread.start()
        downloaded_path = self._do_download(model_id, model_dir)
        if call_back and polling_thread:
            polling_event.set()
            polling_thread.join()
        return downloaded_path

    def verify_hub_token(self):
        self._lazy_init()
        return True if self._token else False

class _HuggingfaceDownloader(_HubDownloader):

    def _build_hub_api(self, token):
        from huggingface_hub import HfApi
        return HfApi(token=token)

    def _verify_hub_token(self, token):
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.whoami(token)
            return True
        except Exception:
            if token: lazyllm.LOG.warning(f'Huggingface token {token} verified failed')
            return False

    def _verify_model_id_impl(self, model_id):
        try:
            self._api.model_info(model_id)
            return True
        except Exception as e:
            lazyllm.LOG.warning('Verify failed: ', e)
            return False

    def _do_download_impl(self, model_id, model_dir):
        from huggingface_hub import snapshot_download
        # refer to https://huggingface.co/docs/huggingface_hub/v0.23.1/en/package_reference/file_download
        downloaded_path = snapshot_download(repo_id=model_id, local_dir=model_dir, token=self._token)
        lazyllm.LOG.info(f'model downloaded at {downloaded_path}')
        return downloaded_path

    def _get_repo_files_impl(self, model_id):
        assert self._api
        orgin_info = self._api.list_repo_tree(model_id, expand=True, recursive=True)
        hub_model_info = []
        for item in list(orgin_info):
            if hasattr(item, 'size'):
                hub_model_info.append({
                    'Path': item.path,
                    'Size': item.size,
                    'SHA': item.blob_id,
                })
        return hub_model_info

class _ModelscopeDownloader(_HubDownloader):

    def _build_hub_api(self, token):
        from modelscope.hub.api import HubApi
        api = HubApi()
        if token:
            api.login(token)
        return api

    def _verify_hub_token(self, token):
        from modelscope.hub.api import HubApi
        api = HubApi()
        try:
            api.login(token)
            return True
        except Exception:
            if token: lazyllm.LOG.warning(f'Modelscope token {token} verified failed')
            return False

    def _verify_model_id_impl(self, model_id):
        try:
            self._api.get_model(model_id)
            return True
        except Exception as e:
            lazyllm.LOG.warning('Verify failed: ', e)
            return False

    def _do_download_impl(self, model_id, model_dir):
        from modelscope.hub.snapshot_download import snapshot_download
        # refer to https://www.modelscope.cn/docs/models/download
        downloaded_path = snapshot_download(model_id=model_id, local_dir=model_dir)
        lazyllm.LOG.info(f'Model downloaded at {downloaded_path}')
        return downloaded_path

    def _get_repo_files_impl(self, model_id):
        assert self._api
        orgin_info = self._api.get_model_files(model_id, recursive=True)
        hub_model_info = []
        for item in orgin_info:
            if item['Type'] == 'blob':
                hub_model_info.append({
                    'Path': item['Path'],
                    'Size': item['Size'],
                    'SHA': item['Sha256']
                })
        return hub_model_info

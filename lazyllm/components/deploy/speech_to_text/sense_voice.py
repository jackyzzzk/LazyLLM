import os
import importlib.util

import lazyllm
from lazyllm.components.utils.file_operate import _base64_to_file, _is_base64_with_mime
from lazyllm import LOG, LazyLLMLaunchersBase, is_valid_url, is_valid_path
from ..base import LazyLLMDeployBase
from ...utils.downloader import ModelManager
from lazyllm.thirdparty import funasr
from typing import Optional

supported_formats = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma')

class SenseVoice(object):
    """SenseVoice 类，封装了基于 FunASR 的语音转文本模型加载与调用逻辑。  
支持懒加载、自动模型下载，输入可为字符串路径、URL 或包含音频的字典。  

Args:
    base_path (str): 模型路径或标识符，将通过 ModelManager 下载到本地。  
    source (Optional[str]): 模型来源，若未指定则使用 ``lazyllm.config['model_source']``。  
    init (bool): 是否在初始化时立即加载模型，默认为 ``False``。  

Attributes:
    base_path (str): 下载或解析后的模型路径。  
    model (Optional[funasr.AutoModel]): FunASR 语音识别模型实例，初始化后可用。  
    init_flag: 用于懒加载的标志，确保模型只加载一次。  
"""
    def __init__(self, base_path, source=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_stt)

    def load_stt(self):
        """初始化并加载 FunASR 语音转文本模型，如果存在 `torch_npu` 则支持华为 NPU 加速。

使用 `fsmn-vad` 进行语音活动检测（VAD），支持长语音段。
单段语音最大持续时间为 30 秒。
默认推理设备为 `cuda:0`（GPU）。

加载的模型将保存在 `self.model` 中，用于后续音频转写。

注意：
- 如果当前环境中存在 `torch_npu`，函数会导入以支持昇腾 NPU 加速。
"""
        if importlib.util.find_spec('torch_npu') is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        self.model = funasr.AutoModel(
            model=self.base_path,
            trust_remote_code=False,
            vad_model='fsmn-vad',
            vad_kwargs={'max_single_segment_time': 30000},
            device='cuda:0',
        )

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_stt)
        if isinstance(string, dict):
            if string['audio']:
                string = string['audio'][-1] if isinstance(string['audio'], list) else string['audio']
            else:
                string = string['inputs']
        assert isinstance(string, str)
        string = string.strip()
        try:
            string = _base64_to_file(string) if _is_base64_with_mime(string) else string
        except Exception as e:
            LOG.error(f'Error processing base64 encoding: {e}')
            return f'Error processing base64 encoding {e}'
        if not string.endswith(supported_formats):
            return f'Only {", ".join(supported_formats)} formats in the form of file paths or URLs are supported.'
        if not is_valid_path(string) and not is_valid_url(string):
            return f'This {string} is not a valid URL or file path. Please check.'
        res = self.model.generate(
            input=string,
            cache={},
            language='auto',  # 'zn', 'en', 'yue', 'ja', 'ko', 'nospeech'
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = funasr.utils.postprocess_utils.rich_transcription_postprocess(res[0]['text'])
        return text

    @classmethod
    def rebuild(cls, base_path, init):
        """类方法，用于在反序列化过程中重新构建 `SenseVoice` 实例（例如使用 `cloudpickle`）。  

Args:
    base_path (str): 语音识别模型路径。  
    init (bool): 实例化时是否立即初始化并加载模型。

**Returns:**

- SenseVoice: 返回一个新的 `SenseVoice` 实例，用于支持序列化/多进程兼容。
"""
        return cls(base_path, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return SenseVoice.rebuild, (self.base_path, init)

class SenseVoiceDeploy(LazyLLMDeployBase):
    """SenseVoice 模型部署类。该类用于将SenseVoice模型部署到指定服务器上，以便可以通过网络进行调用。

`__init__(self, launcher=None)`
构造函数，初始化部署类。

Args:
    launcher (Optional[LazyLLMLaunchersBase]): Launcher instance, defaults to None.
    log_path (Optional[str]): Log file path, defaults to None.
    trust_remote_code (bool): Whether to trust remote code, defaults to True.
    port (Optional[int]): Service port number, defaults to None.

Notes:
    - 推理的输入：字符串。音频路径或者链接。
    - 推理的返回值：字符串。识别出的内容。
    - 支持的模型为：[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)


Examples:
    >>> import os
    >>> import lazyllm
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import SenseVoiceDeploy
    >>> deployer = SenseVoiceDeploy(launchers.remote())
    >>> url = deployer(base_model='SenseVoiceSmall')
    >>> model = UrlModule(url=url)
    >>> model('path/to/audio') # support format: .mp3, .wav
    ... xxxxxxxxxxxxxxxx
    """
    keys_name_handle = {
        'inputs': 'inputs',
        'audio': 'audio',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'audio': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher: Optional[LazyLLMLaunchersBase] = None,
                 log_path: Optional[str] = None, trust_remote_code: bool = True, port: Optional[int] = None, **kw):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.pt', '.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                        f'base_model({base_model}) will be used')
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(port=self._port, func=SenseVoice(finetuned_model), launcher=self._launcher,
                                          log_path=self._log_path, cls='sensevoice')()

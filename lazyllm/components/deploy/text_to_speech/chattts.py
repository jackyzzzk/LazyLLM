from lazyllm.thirdparty import torch, ChatTTS
from .utils import TTSBase
from .base import _TTSInfer


class _ChatTTSModule(_TTSInfer):

    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True):
        self.seed = 1024
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'chattts')
        raise RuntimeError('ChatTTS is deprecated and no longer supported.')

    def _load_model(self):
        self.model = ChatTTS.Chat()
        self.model.load(compile=False,
                        source='custom',
                        custom_path=self.base_path)
        self.spk = self._set_spk(self.seed)

    def _set_spk(self, seed):
        assert self.model
        torch.manual_seed(seed)
        rand_spk = self.model.sample_random_speaker()
        return rand_spk

    def _infer(self, string):
        if isinstance(string, str):
            query = string
            params_refine_text = ChatTTS.Chat.RefineTextParams()
            params_infer_code = ChatTTS.Chat.InferCodeParams(spk_emb=self.spk)
        elif isinstance(string, dict):
            query = string['inputs']
            params_refine_text = ChatTTS.Chat.RefineTextParams(**string['refinetext'])
            spk_seed = string['infercode']['spk_emb']
            spk_seed = int(spk_seed) if spk_seed else spk_seed
            if isinstance(spk_seed, int) and self.seed != spk_seed:
                self.seed = spk_seed
                self.spk = self._set_spk(self.seed)
            string['infercode']['spk_emb'] = self.spk
            params_infer_code = ChatTTS.Chat.InferCodeParams(**string['infercode'])
        else:
            raise TypeError(f'Not support input type:{type(string)}, requires str or dict.')
        speech = self.model.infer(query,
                                  params_refine_text=params_refine_text,
                                  params_infer_code=params_infer_code,
                                )
        return speech, self.sample_rate

class ChatTTSDeploy(TTSBase):
    """ChatTTS 模型部署类。

Keyword Args: 
    keys_name_handle (dict): 键名映射字典，用于处理内部和外部API接口之间的参数名称转换。
                            默认为 `{'inputs': 'inputs'}`。

    message_format (dict): 请求负载结构，包含三个主要部分：

        - `inputs` (str): 要合成为语音的原始文本内容。

        - `refinetext` (dict): 文本细化和风格化参数，控制语音表达：

            * `prompt` (str): 语音风格控制标签，例如："[oral_2][laugh_0][break_6]"

            * `top_P` (float): 核采样参数，用于解码策略（默认值：0.7）

            * `top_K` (int): Top-K 采样参数（默认值：20）

            * `temperature` (float): 采样温度，控制随机性（默认值：0.7）

            * `repetition_penalty` (float): 重复惩罚，避免冗余生成（默认值：1.0）

            * `max_new_token` (int): 最大生成token数（默认值：384）

            * `min_new_token` (int): 最小生成token数（默认值：0）

            * `show_tqdm` (bool): 是否在生成过程中显示进度条（默认值：True）

            * `ensure_non_empty` (bool): 确保生成非空结果（默认值：True）

        - `infercode` (dict): 推理和编码参数，影响音频质量：

            * `prompt` (str): 语速控制标签，例如："[speed_5]"

            * `spk_emb` (可选): 说话人嵌入向量，用于指定音色特征（默认值：None）

            * `temperature` (float): 音频生成的采样温度（默认值：0.3）

            * `repetition_penalty` (float): 重复惩罚系数（默认值：1.05）

            * `max_new_token` (int): 音频生成的最大token数（默认值：2048）



Examples:
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import ChatTTSDeploy
    >>> deployer = ChatTTSDeploy(launchers.remote())
    >>> url = deployer(base_model='ChatTTS')
    >>> model = UrlModule(url=url)
    >>> res = model('Hello World!')
    >>> print(res)
    ... <lazyllm-query>{"query": "", "files": ["path/to/chattts/sound_xxx.wav"]}
    """
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'refinetext': {
            'prompt': '[oral_2][laugh_0][break_6]',
            'top_P': 0.7,
            'top_K': 20,
            'temperature': 0.7,
            'repetition_penalty': 1.0,
            'max_new_token': 384,
            'min_new_token': 0,
            'show_tqdm': True,
            'ensure_non_empty': True,
        },
        'infercode': {
            'prompt': '[speed_5]',
            'spk_emb': None,
            'temperature': 0.3,
            'repetition_penalty': 1.05,
            'max_new_token': 2048,
        }

    }
    default_headers = {'Content-Type': 'application/json'}
    func = _ChatTTSModule

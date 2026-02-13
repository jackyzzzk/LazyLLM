import os
import sys
import json
import random
import importlib.metadata
from packaging.version import parse
from typing import Optional

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG, LazyLLMLaunchersBase
from .base import LazyLLMDeployBase, verify_fastapi_func, verify_func_factory
from ...common import LazyLLMRegisterMetaClass
from .utils import get_log_path, make_log_dir, parse_options_keys
from .ray import reallocate_launcher, Distributed, sleep_moment


verify_vllm_openai_func = verify_func_factory('ERROR:', 'Application startup complete.')

class _VllmStreamParseParametersMeta(LazyLLMRegisterMetaClass):
    def __getattribute__(cls, name):
        if name == 'stream_parse_parameters':
            if not hasattr(cls, '_stream_parse_parameters'):
                try:
                    vllm_version = parse(importlib.metadata.version('vllm'))
                    cls._stream_parse_parameters = {'decode_unicode': False}
                    if vllm_version <= parse('0.5.0'): cls._stream_parse_parameters.update({'delimiter': b'\0'})
                except ImportError:
                    cls._stream_parse_parameters = {'decode_unicode': False}
            return cls._stream_parse_parameters
        return super().__getattribute__(name)


class Vllm(LazyLLMDeployBase, metaclass=_VllmStreamParseParametersMeta):
    """此类是 ``LazyLLMDeployBase`` 的子类，基于 [VLLM](https://github.com/vllm-project/vllm) 框架提供的推理能力，用于大语言模型的部署与推理。

Args:
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 模型启动器，默认为 ``launchers.remote(ngpus=1)``。
    log_path (str): 日志保存路径，若为 ``None`` 则不保存日志。
    openai_api (bool): 是否使用 OpenAI API 接口启动 VLLM 服务，默认为 ``False``。
    kw: 关键字参数，用于更新默认的部署参数。除支持的关键字参数外，不允许传入额外参数。

此类支持的关键字参数及其默认值如下：

Keyword Args: 
    tensor-parallel-size (int): 张量并行大小，默认为 ``1``。
    dtype (str): 模型权重和激活值的数据类型，默认为 ``auto``。可选：``half``、``float16``、``bfloat16``、``float``、``float32``。
    kv-cache-dtype (str): KV 缓存的数据类型，默认为 ``auto``。可选：``fp8``、``fp8_e5m2``、``fp8_e4m3``。
    device (str): VLLM 支持的硬件类型，默认为 ``auto``。可选：``cuda``、``neuron``、``cpu``。
    block-size (int): token 块大小，默认为 ``16``。
    port (int | str): 服务端口号，默认为 ``auto``，即随机分配。
    host (str): 服务绑定的 IP 地址，默认为 ``0.0.0.0``。
    seed (int): 随机数种子，默认为 ``0``。
    tokenizer_mode (str): Tokenizer 加载模式，默认为 ``auto``。
    max-num-seqs (int): 推理引擎支持的最大并行请求数，默认为 ``256``。
    pipeline-parallel-size (int): 流水线并行大小，默认为 ``1``。
    max-num-batched-tokens (int): 最大批处理 token 数，默认为 ``64000``。


Examples:
    >>> from lazyllm import deploy
    >>> infer = deploy.vllm()
    """
    # keys_name_handle/default_headers/message_format will lose efficacy when openai_api is True
    keys_name_handle = {'inputs': 'prompt', 'stop': 'stop'}
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'stop': ['<|im_end|>', '<|im_start|>', '</s>', '<|assistant|>', '<|user|>', '<|system|>', '<eos>'],
        'skip_special_tokens': False,
        'temperature': 0.6,
        'top_p': 0.8,
        'max_tokens': 4096
    }
    auto_map = {
        'tp': 'tensor_parallel_size'
    }  # from cli to vllm
    optional_keys = set([
        'max_model_len',
        'gpu_memory_utilization',
        'task',
        'dtype',
        'kv_cache_dtype',
        'tokenizer_mode',
        'block_size',
        'max_num_seqs',
        'pipeline_parallel_size',
        'tensor_parallel_size',
        'seed',
        'port',
        'max_num_batched_tokens',
        'tool_call_parser',
        'swap_space',
        'mm_processor_kwargs',
        'limit_mm_per_prompt',
        'hf_overrides'])

    # TODO(wangzhihong): change default value for `openai_api` argument to True
    def __init__(self, trust_remote_code: bool = True, launcher: LazyLLMLaunchersBase = launchers.remote(ngpus=1),  # noqa B008
                 log_path: str = None, openai_api: Optional[bool] = None, **kw):
        self.launcher_list, launcher = reallocate_launcher(launcher)
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'host': '0.0.0.0',
            'max_model_len': 10240,
        })
        if openai_api is None: openai_api = lazyllm.config['openai_api']
        self._vllm_cmd = 'vllm.entrypoints.openai.api_server' if openai_api else 'vllm.entrypoints.api_server'
        self._openai_api = openai_api
        self.options_keys = kw.pop('options_keys', [])
        if trust_remote_code and 'trust_remote_code' not in self.options_keys:
            self.options_keys.append('trust_remote_code')
        self.kw.update(**{key: kw[key] for key in self.optional_keys if key in kw})
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'vllm') if log_path else None
        if self.launcher_list:
            ray_launcher = [Distributed(launcher=launcher) for launcher in self.launcher_list]
            parall_launcher = [lazyllm.pipeline(sleep_moment, launcher) for launcher in ray_launcher[1:]]
            self._prepare_deploy = lazyllm.pipeline(
                ray_launcher[0], post_action=(lazyllm.parallel(*parall_launcher) if len(parall_launcher) else None))

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """构造用于启动 vLLM 推理服务的命令。

该方法会自动检测模型路径是否有效，并根据当前配置参数动态生成可执行命令，支持多节点部署时自动加入 ray 启动命令。

Args:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 备用基础模型路径（当 finetuned_model 无效时启用）。
    master_ip (str): 分布式部署中的主节点 IP，仅在多节点时启用。

**Returns:**

- LazyLLMCMD: 可执行命令对象，包含启动指令、结果回调函数及健康检查方法。
"""
        if finetuned_model:
            LOG.info(f'Using finetuned model from {finetuned_model} to deploy.')
        if not finetuned_model:
            LOG.info(f'Using model {base_model} to deploy.')
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model):
            LOG.warning(f'Warning! The finetuned_model path does not exist: {finetuned_model}. '
                        f'Using base_model({base_model}) instead.')
            finetuned_model = base_model
        elif not any(filename.endswith(('.bin', '.safetensors', '.pt'))
                     for filename in os.listdir(finetuned_model)):
            LOG.warning(f'Warning! No valid model files (.bin, .safetensors or .pt) found in: {finetuned_model}. '
                        f'Using base_model({base_model}) instead.')
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            cmd = ''
            if self.launcher_list:
                cmd += f'ray start --address="{master_ip}" && '
            cmd += f'{sys.executable} -m {self._vllm_cmd} --model {finetuned_model} '
            if self._openai_api: cmd += '--served-model-name lazyllm '
            cmd += self.kw.parse_kwargs()
            cmd += ' ' + parse_options_keys(self.options_keys)
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl,
                          checkf=(verify_vllm_openai_func if self._openai_api else verify_fastapi_func))

    def geturl(self, job=None):
        """获取 vLLM 服务的推理地址。

根据运行模式（Display 模式或实际部署）返回相应的 URL，用于访问模型的生成接口。

Args:
    job (Job, optional): 部署任务对象。默认取当前模块绑定的 job。

**Returns:**

- str: 推理服务的 HTTP 地址。
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/generate'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}' + (
                '/v1/' if self._openai_api else '/generate')

    @staticmethod
    def extract_result(x, inputs):
        """从 VLLM 接口返回的 JSON 字符串中提取推理结果。

Args:
    x (str): VLLM 服务返回的原始 JSON 字符串。
    inputs (Any): 输入参数（此处未使用，保留接口一致性）。

**Returns:**

- str: 模型生成的文本结果。
"""
        return json.loads(x)['text'][0]

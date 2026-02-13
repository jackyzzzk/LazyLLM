import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import make_log_dir, get_log_path, parse_options_keys
from typing import Optional


class Lightllm(LazyLLMDeployBase):
    """此类是 ``LazyLLMDeployBase`` 的子类，基于 [LightLLM](https://github.com/ModelTC/lightllm) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    trust_remote_code (bool, optional): 是否信任远程代码，默认为True
    launcher (Launcher, optional): 任务启动器，默认为单GPU远程启动器
    log_path (str, optional): 日志文件路径，默认为None
    **kw: 其他LightLLM服务器配置参数
此类的关键字参数及其默认值如下：

Keyword Args: 
    tp (int): 张量并行参数，默认为 ``1``。
    max_total_token_num (int): 最大总token数，默认为 ``64000``。
    eos_id (int): 结束符ID，默认为 ``2``。
    port (int): 服务的端口号，默认为 ``None``。此情况下LazyLLM会自动生成随机端口号。
    host (str): 服务的IP地址，默认为 ``0.0.0.0``。
    nccl_port (int): NCCL 端口，默认为 ``None``。此情况下LazyLLM会自动生成随机端口号。
    tokenizer_mode (str): tokenizer的加载模式，默认为 ``auto``。
    running_max_req_size (int): 推理引擎最大的并行请求数， 默认为 ``256``。
    data_type (str): 模型权重的数据类型，默认为 ``float16``。
    max_req_total_len (int): 请求的最大总长度，默认为 ``64000``。
    max_req_input_len (int): 输入的最大长度，默认为 ``4096``。
    long_truncation_mode (str): 长文本的截断模式，默认为 ``head``。


Examples:
    >>> from lazyllm import deploy
    >>> infer = deploy.lightllm()
    """
    keys_name_handle = {
        'inputs': 'inputs',
        'stop': 'stop_sequences'
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'inputs': 'Who are you ?',
        'parameters': {
            'do_sample': False,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'repetition_penalty': 1.0,
            'temperature': 1.0,
            'top_p': 1,
            'top_k': -1,  # -1 is for all
            'ignore_eos': False,
            'max_new_tokens': 8192,
            'stop_sequences': None,
        }
    }
    auto_map = {}
    stream_url_suffix = '_stream'
    stream_parse_parameters = {'delimiter': b'\n\n'}

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(ngpus=1), log_path=None, # noqa B008
                 openai_api: Optional[bool] = None, **kw):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'tp': 1,
            'max_total_token_num': 64000,
            'eos_id': 2,
            'port': None,
            'host': '0.0.0.0',
            'nccl_port': None,
            'tokenizer_mode': 'auto',
            'running_max_req_size': 256,
            'data_type': 'float16',
            'max_req_total_len': 64000,
            'max_req_input_len': 4096,
            'long_truncation_mode': 'head',
        })
        self.options_keys = kw.pop('options_keys', [])
        if trust_remote_code and 'trust_remote_code' not in self.options_keys:
            self.options_keys.append('trust_remote_code')
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] else True
        self.random_nccl_port = False if 'nccl_port' in kw and kw['nccl_port'] else True
        self.temp_folder = make_log_dir(log_path, 'lightllm') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        """该方法用于生成启动LightLLM服务的命令。

Args:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 基础模型路径，当finetuned_model无效时使用。

**Returns:**

- LazyLLMCMD: 一个包含启动命令的LazyLLMCMD对象。
"""
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)
            if self.random_nccl_port:
                self.kw['nccl_port'] = random.randint(20000, 30000)
            cmd = f'python -m lightllm.server.api_server --model_dir {finetuned_model} '
            cmd += self.kw.parse_kwargs()
            cmd += ' ' + parse_options_keys(self.options_keys)
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """获取LightLLM服务的URL地址。

Args:
    job (optional): 任务对象，默认为None，此时使用self.job。

**Returns:**

- str: 服务的URL地址，格式为"http://{ip}:{port}/generate"。
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/generate'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x, inputs):
        """从服务响应中提取生成的文本结果。

Args:
    x (str): 服务返回的响应文本。
    inputs (str): 输入文本。

**Returns:**

- str: 提取出的生成文本。

异常:
    Exception: 当解析JSON响应失败时抛出异常。
"""
        try:
            if x.startswith('data:'): return json.loads(x[len('data:'):])['token']['text']
            else: return json.loads(x)['generated_text'][0]
        except Exception as e:
            LOG.warning(f'JSONDecodeError on load {x}')
            raise e

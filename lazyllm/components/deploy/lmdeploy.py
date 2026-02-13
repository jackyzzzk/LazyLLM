import os
import json
import random
import importlib.util

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG, config
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import get_log_path, make_log_dir, parse_options_keys


config.add('lmdeploy_eager_mode', bool, False, 'LMDEPLOY_EAGER_MODE',
           description='Whether to use eager mode for lmdeploy.')

class LMDeploy(LazyLLMDeployBase):
    """``LMDeploy`` 类，继承自 ``LazyLLMDeployBase``，基于 [LMDeploy](https://github.com/InternLM/lmdeploy) 框架，  
用于启动并管理大语言模型的推理服务。

Args:
    launcher (Optional[lazyllm.launcher]): 服务启动器，默认使用 ``launchers.remote(ngpus=1)``。  
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。  
    log_path (Optional[str]): 日志输出路径，默认为 ``None``。  
    **kw: 关键字参数，用于更新默认的部署配置。除下列参数外，不允许传入额外参数。  

Keyword Args:
    tp (int): 张量并行参数，默认为 ``1``。  
    server-name (str): 服务监听的 IP 地址，默认为 ``0.0.0.0``。  
    server-port (Optional[int]): 服务端口号，默认为 ``None``，此时会自动随机分配 30000–40000 区间的端口。  
    max-batch-size (int): 最大批处理大小，默认为 ``128``。  
    chat-template (Optional[str]): 对话模板文件路径。若模型不是视觉语言模型且未指定模板，将使用默认模板。  
    eager-mode (bool): 是否启用 eager 模式，受环境变量 ``LMDEPLOY_EAGER_MODE`` 控制，默认为 ``False``。  


Examples:
    >>> # Basic use:
    >>> from lazyllm import deploy
    >>> infer = deploy.LMDeploy()
    >>>
    >>> # MultiModal:
    >>> import lazyllm
    >>> from lazyllm import deploy, globals
    >>> from lazyllm.components.formatter import encode_query_with_filepaths
    >>> chat = lazyllm.TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)
    >>> chat.update_server()
    >>> inputs = encode_query_with_filepaths('What is it?', ['path/to/image'])
    >>> res = chat(inputs)
    """
    keys_name_handle = {
        'inputs': 'prompt',
        'stop': 'stop',
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'stop': None,
        'top_p': 0.8,
        'temperature': 0.8,
        'skip_special_tokens': True,
    }
    auto_map = {
        'port': 'server-port',
        'host': 'server-name',
        'max_batch_size': 'max-batch-size',
    }
    stream_parse_parameters = {'delimiter': b'\n'}

    def __init__(self, launcher=launchers.remote(ngpus=1), trust_remote_code=True, log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'server-name': '0.0.0.0',
            'server-port': None,
            'tp': 1,
            'max-batch-size': 128,
        })
        self.options_keys = kw.pop('options_keys', [])
        self.kw.check_and_update(kw)
        self._trust_remote_code = trust_remote_code
        self.random_port = False if 'server-port' in kw and kw['server-port'] else True
        self.temp_folder = make_log_dir(log_path, 'lmdeploy') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        """该方法用于生成启动LMDeploy服务的命令。

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
                self.kw['server-port'] = random.randint(30000, 40000)
            cmd = f'lmdeploy serve api_server {finetuned_model} --model-name lazyllm '

            if importlib.util.find_spec('torch_npu') is not None: cmd += '--device ascend '
            if config['lmdeploy_eager_mode']: cmd += '--eager-mode '
            cmd += self.kw.parse_kwargs()
            cmd += ' ' + parse_options_keys(self.options_keys)
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """获取LMDeploy服务的URL地址。

Args:
    job (optional): 任务对象，默认为None，此时使用self.job。

**Returns:**

- str: 服务的URL地址，格式为"http://{ip}:{port}/v1/chat/interactive"。
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/v1/'
        else:
            return f'http://{job.get_jobip()}:{self.kw["server-port"]}/v1/'

    @staticmethod
    def extract_result(x, inputs):
        """解析模型推理结果，从返回的 JSON 字符串中提取文本输出。

Args:
    x (str): 模型返回的 JSON 格式字符串。  
    inputs (dict): 原始输入数据（此参数未被直接使用，保留作接口兼容）。  

**Returns:**

- str: 从响应中解析得到的文本结果。  
"""
        return json.loads(x)['text']

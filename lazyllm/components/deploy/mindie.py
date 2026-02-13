import os
import json
import random
import shutil

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_func_factory
from .utils import get_log_path, make_log_dir

lazyllm.config.add('mindie_home', str, '', 'MINDIE_HOME', description='The home directory of MindIE.')

verify_fastapi_func = verify_func_factory(error_message='Service Startup Failed',
                                          running_message='Daemon start success')
class Mindie(LazyLLMDeployBase):
    """此类是 ``LazyLLMDeployBase`` 的一个子类, 用于部署和管理MindIE大模型推理服务。它封装了MindIE服务的配置生成、进程启动和API交互的全流程。

Args:
    trust_remote_code (bool): 是否信任远程代码(如HuggingFace模型)。默认为 ``True``。
    launcher: 任务启动器实例，默认为 ``launchers.remote()``。
    log_path (str): 日志保存路径，若为 ``None`` 则不保存日志。
    **kw: 其他配置参数

Keyword Args: 
            npuDeviceIds: NPU设备ID列表(如 ``[[0,1]]`` 表示使用2张卡)
            worldSize: 模型并行数量
            port: 服务端口（设为 ``'auto'`` 时自动分配30000-40000的随机端口)
            maxSeqLen: 最大序列长度
            maxInputTokenLen: 单次输入最大token数
            maxPrefillTokens: 预填充token上限
            config: 自定义配置文件

Notes
                : 
   必须预先设置环境变量 ``LAZYLLM_MINDIE_HOME`` 指向MindIE安装目录, 若未指定 ``finetuned_model`` 或路径无效，会自动回退到 ``base_model``


Examples:
    >>> import lazyllm
    >>> from lazyllm.components.deploy import Mindie            
    >>> deployer = Mindie(
    ...     port=30000,
    ...     launcher=lazyllm.launchers.remote(),
    ...     max_seq_len=32000,
    ...     log_path="/path/to/logs"
    ... )
    >>> cmd = deployer.cmd(
    ...     finetuned_model="/path/to/finetuned_model",
    ...     base_model="/path/to/base_model")
    >>> print("Service URL:", cmd.geturl())

    """
    keys_name_handle = {
        'inputs': 'prompt',
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'max_tokens': 4096,
        'presence_penalty': 1.03,
        'frequency_penalty': 1.0,
        'temperature': 0.5,
        'top_p': 0.95
    }
    auto_map = {
        'port': int,
        'tp': ('worldSize', int),
        'max_input_token_len': ('maxInputTokenLen', int),
        'max_prefill_tokens': ('maxPrefillTokens', int),
        'max_seq_len': ('maxSeqLen', int)
    }

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(), log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        assert lazyllm.config['mindie_home'], 'Ensure you have installed MindIE and \
                                  "export LAZYLLM_MINDIE_HOME=/path/to/mindie/latest"'
        self.mindie_home = lazyllm.config['mindie_home']
        self.mindie_config_path = os.path.join(self.mindie_home, 'mindie-service/conf/config.json')
        self.backup_path = self.mindie_config_path + '.backup'
        self.custom_config = kw.pop('config', None)
        self.kw = ArgsDict({
            'npuDeviceIds': [[0]],
            'worldSize': 1,
            'port': 'auto',
            'host': '0.0.0.0',
            'maxSeqLen': 64000,
            'maxInputTokenLen': 4096,
            'maxPrefillTokens': 8192,
        })
        self.trust_remote_code = trust_remote_code
        self.options_keys = kw.pop('options_keys', [])
        assert len(self.options_keys) == 0, 'options_keys is not supported'
        kw.update({'skip_check': False})
        self.kw.check_and_update(kw)
        self.kw['npuDeviceIds'] = [[i for i in range(self.kw.get('worldSize', 1))]]
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'mindie') if log_path else None

        if self.custom_config:
            self.config_dict = (ArgsDict(self.load_config(self.custom_config))
                                if isinstance(self.custom_config, str) else ArgsDict(self.custom_config))
            self.kw['host'] = self.config_dict['ServerConfig']['ipAddress']
            self.kw['port'] = self.config_dict['ServerConfig']['port']
        else:
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindie', 'config.json')
            self.config_dict = ArgsDict(self.load_config(default_config_path))

    def __del__(self):
        if hasattr(self, 'backup_path') and os.path.isfile(self.backup_path):
            shutil.copy2(self.backup_path, self.mindie_config_path)

    def load_config(self, config_path):
        """加载并解析MindIE配置文件。

Args:
    config_path (str): JSON配置文件的路径

**Returns:**

- dict: 解析后的配置字典

注意事项:
    - 处理默认和自定义配置文件
    - 使用JSON格式配置
    - 修改前会创建原始配置的备份
"""
        with open(config_path, 'r') as file:
            config_dict = json.load(file)
        return config_dict

    def save_config(self):
        """保存当前配置到文件。

注意事项:
    - 自动创建现有配置的备份
    - 写入到标准MindIE配置位置
    - 使用带缩进的JSON格式
    - 部署时自动调用
"""
        if os.path.isfile(self.mindie_config_path):
            shutil.copy2(self.mindie_config_path, self.backup_path)

        with open(self.mindie_config_path, 'w') as file:
            json.dump(self.config_dict, file)

    def update_config(self):
        """使用当前设置更新配置字典。

注意事项:
    - 处理多个配置部分:
        - 模型部署参数
        - 服务器设置
        - 调度参数
"""
        backend_config = self.config_dict['BackendConfig']
        backend_config['npuDeviceIds'] = self.kw['npuDeviceIds']
        model_config = {
            'modelName': self.finetuned_model.split('/')[-1],
            'modelWeightPath': self.finetuned_model,
            'worldSize': self.kw['worldSize'],
            'trust_remote_code': self.trust_remote_code
        }
        backend_config['ModelDeployConfig']['ModelConfig'][0].update(model_config)
        backend_config['ModelDeployConfig']['maxSeqLen'] = self.kw['maxSeqLen']
        backend_config['ModelDeployConfig']['maxInputTokenLen'] = self.kw['maxInputTokenLen']
        backend_config['ScheduleConfig']['maxPrefillTokens'] = self.kw['maxPrefillTokens']
        self.config_dict['BackendConfig'] = backend_config
        if self.kw['host'] != '0.0.0.0':
            self.config_dict['ServerConfig']['ipAddress'] = self.kw['host']
        self.config_dict['ServerConfig']['port'] = self.kw['port']

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """生成启动MindIE服务的命令。

Args:
    finetuned_model (str): 微调模型路径
    base_model (str): 基础模型路径(当微调模型无效时作为后备)
    master_ip (str): 主节点IP地址(当前未使用)

**Returns:**

- LazyLLMCMD: 启动服务的命令对象

注意事项:
    - 自动处理模型路径验证
    - 启动服务前更新配置
    - 支持配置随机端口分配
"""
        if self.custom_config is None:
            self.finetuned_model = finetuned_model
            if finetuned_model or base_model:
                if not os.path.exists(finetuned_model) or \
                    not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                            for filename in os.listdir(finetuned_model)):
                    if not finetuned_model:
                        LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                                    f'base_model({base_model}) will be used')
                    self.finetuned_model = base_model

            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            self.update_config()

        self.save_config()

        def impl():
            cmd = f'{os.path.join(self.mindie_home, "mindie-service/bin/mindieservice_daemon")}'
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """获取部署后的服务URL。

Args:
    job: 任务对象(可选，默认为self.job)

**Returns:**

- str: generate接口的URL

注意事项:
    - 根据显示模式返回不同格式
    - 包含配置中的端口号
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'
        else:
            LOG.info(f'MindIE Server running on http://{job.get_jobip()}:{self.kw["port"]}')
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x, inputs):
        """从API响应中提取生成的文本。

Args:
    x: 原始API响应
    inputs: 原始输入(未使用)

**Returns:**

- str: 生成的文本

注意事项:
    - 解析JSON响应
    - 返回响应中的第一个文本条目
"""
        return json.loads(x)['text'][0]

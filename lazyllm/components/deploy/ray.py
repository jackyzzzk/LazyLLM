import time
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD
from .base import LazyLLMDeployBase


lazyllm.config.add('num_gpus_per_node', int, 8, 'NUM_GPUS_PER_NODE',
                   description='The number of GPUs per node for Ray launcher when deploy models.')

def sleep_moment(finetuned_model=None, base_model=None, master_ip=None):
    sleep_time = random.uniform(1, 3)
    time.sleep(sleep_time)
    return lazyllm.package(finetuned_model, base_model, master_ip)

def reallocate_launcher(launcher):
    if not isinstance(launcher, (launchers.ScoLauncher, launchers.SlurmLauncher)):
        return [], launcher
    nnode = launcher.nnode
    ngpus = launcher.ngpus
    if nnode <= 1 and ngpus <= lazyllm.config['num_gpus_per_node']:
        return [], launcher
    if nnode > 1 and ngpus <= lazyllm.config['num_gpus_per_node']:
        return [launcher.__class__(ngpus=ngpus, sync=False) for _ in range(nnode - 1)], \
            launcher.__class__(ngpus=ngpus, sync=False)
    else:
        erro_info = (
            f'At least 1 node is required. The number of GPUs({ngpus}) in a single node exceeds the upper '
            f'limit{(lazyllm.config["num_gpus_per_node"])}. Please check the actual '
            'number of GPUs in a single node and set the environment variable: LAZYLLM_NUM_GPUS_PER_NODE.')
        lazyllm.LOG.error(erro_info)
        raise RuntimeError(erro_info)

class Distributed(LazyLLMDeployBase):
    """分布式部署类，继承自LazyLLMDeployBase。

提供基于Ray框架的分布式模型部署功能，支持多节点集群部署。

Args:
    launcher: 启动器配置，默认为远程启动器(ngpus=1)
    port (int, optional): 服务端口号，默认为随机端口(30000-40000)

Attributes:
    finetuned_model: 微调后的模型路径
    base_model: 基础模型路径
    master_ip: 主节点IP地址

Methods:
    cmd(finetuned_model, base_model, master_ip): 生成部署命令
    geturl(job): 获取部署服务的URL地址
"""

    def __init__(self, launcher=launchers.remote(ngpus=1), port=None):  # noqa B008
        super().__init__(launcher=launcher)
        self.port = port or random.randint(30000, 40000)
        self.finetuned_model = None
        self.base_model = None
        self.master_ip = None

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """生成Ray分布式部署命令。

根据是否为主节点生成相应的Ray启动命令，支持头节点和工作节点两种模式。

Args:
    finetuned_model: 微调后的模型路径
    base_model: 基础模型路径
    master_ip: 主节点IP地址，如果为空则作为头节点启动

Returns:
    LazyLLMCMD: 包含部署命令的对象
"""
        self.finetuned_model = finetuned_model
        self.base_model = base_model
        self.master_ip = master_ip
        if not self.master_ip:
            cmd = f'ray start --block --head --port={self.port} && sleep 365d'
        else:
            cmd = f'ray start --address={self.master_ip} && sleep 365d'
        return LazyLLMCMD(cmd=cmd, return_value=self.geturl)

    def geturl(self, job=None):
        """获取分布式部署服务的URL地址。

根据部署模式返回相应的服务地址信息，支持显示模式和实际部署模式。

Args:
    job: 任务对象，默认为当前任务

Returns:
    Package: 包含模型路径和服务地址的打包对象
"""
        time.sleep(5)
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return lazyllm.package(self.finetuned_model, self.base_model, None)
        else:
            if self.master_ip:
                return lazyllm.package(self.finetuned_model, self.base_model, self.master_ip)
            return lazyllm.package(self.finetuned_model, self.base_model, f'{job.get_jobip()}:{self.port}')

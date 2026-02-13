import os
import copy
import json
import random
import subprocess
from datetime import datetime
from subprocess import CalledProcessError

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty, LOG
from .base import LazyLLMFinetuneBase


class EasyR1Finetune(LazyLLMFinetuneBase):
    """此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [EasyR1](https://github.com/hiyouga/EasyR1) 框架提供的强化学习的能力，用于对大语言模型进行强化学习训练。

Args:
    base_model: 用于进行训练的基模型路径。支持本地路径，若路径不存在则尝试从配置的模型路径中查找。
    target_path: 训练完成后，模型权重保存的目标路径。
    merge_path (str, optional): 同 ``target_path``，不需要配置。
    launcher (lazyllm.launcher, optional): 训练任务的启动器，默认为``None``, 使用单卡同步远程启动器 ``launchers.remote(ngpus=1, sync=True)``。
    **kw: 关键字参数，用于动态覆盖默认训练配置中的参数。

此类的关键字参数及其默认值如下：

Keyword Args:
    data.max_prompt_length (int): 默认值是：``2048``。用于限定输入 prompt 的最大 token 长度，超出会根据实现截断或丢弃多余 tokens。
    data.max_response_length (int): 默认值是：``2048``。模型生成时允许的最大响应长度（token 数）。
    data.rollout_batch_size (int): 默认值是：``128``。rollout（策略采样/生成）时的 batch 大小，增大可提高吞吐但占用更多显存。
    data.val_batch_size (int): 默认值是：``1024``。验证/评估阶段的 batch 大小，通常可设大以提高评估效率，但受显存限制。
    data.format_prompt (typing.Optional[typing.Union[str, callable]]): 默认值是：``None``。用于将原始样本格式化为模型输入 prompt 的模板或函数。为 ``None`` 时使用框架/数据集默认格式化逻辑。
    worker.actor.global_batch_size (int): 默认值是：``128``。actor（用于生成与训练的组件）的一次更新对应的全局 batch 大小。
    worker.actor.micro_batch_size_per_device_for_update (int): 默认值是：``4``。训练更新（反向传播）阶段每个设备上的微批大小，用于计算梯度累积步数。
    worker.actor.micro_batch_size_per_device_for_experience (int): 默认值是：``16``。生成经验（rollout）阶段每个设备上的微批大小，通常大于更新阶段以提高采样吞吐。
    worker.rollout.gpu_memory_utilization (float): 默认值是：``0.6``。控制 rollout 阶段每张 GPU 可用显存比例，框架会据此估算可用 batch/并行度以避免 OOM。
    worker.rollout.tensor_parallel_size (int): 默认值是：``1``。rollout/采样阶段使用的 tensor 并行度大小，>1 时启用张量并行以分摊显存与计算。
    worker.reward.reward_function (typing.Optional[typing.Union[str, callable]]): 默认值是：``None``。用于计算 reward 的函数或可识别标识。函数原型通常为 func(samples) -> rewards。自定义 reward 函数须高效且可序列化（或可在子进程/远程环境中调用），以免成为训练瓶颈。
    trainer.total_epochs (int): 默认值是：``2``。训练的总轮次。对于强化学习微调场景通常不需很大轮次，可通过 rollout 次数与 batch 调整训练强度。
    trainer.n_gpus_per_node (int): 默认值是：``1``。每个节点上用于训练的 GPU 数量。
    trainer.save_freq (int): 默认值是：``5``。以 epoch 为单位的 checkpoint 保存频率。设置为 0 或负值的行为由实现决定（可能只在结束时保存）。
    trainer.save_checkpoint_path (typing.Optional[str]): 默认值是：``None``。指定 checkpoint 的保存路径，若 ``None`` 则使用 ``target_path`` 或框架默认路径。
    trainer.save_model_only (bool): 默认值是：``False``。是否仅保存模型权重而不保存优化器/调度器等训练状态。


Examples:
    >>> from lazyllm import finetune
    >>> finetune.easyr1('qwen2-0.5b-instruct', 'path/to/target')
    <lazyllm.llm.finetune type=EasyR1Finetune>
    """
    defatult_kw = ArgsDict({
        'data.max_prompt_length': 2048,
        'data.max_response_length': 2048,
        'data.rollout_batch_size': 128,
        'data.val_batch_size': 1024,
        'data.format_prompt': None,
        'worker.actor.global_batch_size': 128,
        'worker.actor.micro_batch_size_per_device_for_update': 4,
        'worker.actor.micro_batch_size_per_device_for_experience': 16,
        'worker.rollout.gpu_memory_utilization': 0.6,
        'worker.rollout.tensor_parallel_size': 1,
        'worker.reward.reward_function': None,
        'trainer.total_epochs': 2,
        'trainer.n_gpus_per_node': 1,
        'trainer.save_freq': 5,
        'trainer.save_checkpoint_path': None,
        'trainer.save_model_only': False,
    }, with_line=False)

    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 launcher=None,
                 **kw
                 ):
        if not merge_path:
            merge_path = target_path
        os.makedirs(target_path, exist_ok=True)
        os.makedirs(merge_path, exist_ok=True)
        if launcher is None:
            launcher = launchers.remote(ngpus=1, sync=True)
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self._folder_path = os.path.dirname(os.path.abspath(__file__))
        self.kw = copy.deepcopy(self.defatult_kw)
        self.kw.check_and_update(kw)

    def cmd(self, trainset, valset=None) -> str:
        """生成EasyR1训练命令序列。

Args:
    trainset (str): 训练数据集路径(支持相对lazyllm.config['data_path']的路径)
    valset (str, optional): 验证数据集路径

**Returns:**

- str: 完整的shell命令字符串，包含:
    - 训练命令(自动配置参数)
    - 日志重定向(保存到目标路径)
"""
        thirdparty.check_packages(['verl', 'trl'])
        if not os.path.exists(trainset):
            defatult_path = os.path.join(lazyllm.config['data_path'], trainset)
            if os.path.exists(defatult_path):
                trainset = defatult_path
            else:
                raise FileNotFoundError(f'Trainset {trainset} does not exist, please check your path.')
        if not os.path.exists(valset):
            defatult_path = os.path.join(lazyllm.config['data_path'], valset)
            if os.path.exists(defatult_path):
                valset = defatult_path
            else:
                raise FileNotFoundError(f'Valset {valset} does not exist, please check your path.')

        formatted_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_value = random.randint(1000, 9999)
        self.log_file_path = f'{self.target_path}/train_log_{formatted_date}_{random_value}.log'

        self.kw['data.train_files'] = trainset
        self.kw['data.val_files'] = valset
        self.kw['worker.actor.model.model_path'] = self.base_model
        self.kw['trainer.n_gpus_per_node'] = self.launcher.ngpus
        if not self.kw['trainer.save_checkpoint_path']:
            self.kw['trainer.save_checkpoint_path'] = self.target_path
        if not self.kw['worker.reward.reward_function']:
            self.kw['worker.reward.reward_function'] = (f'{self._folder_path}/easy_r1/'
                                                        'reward_function/math.py:compute_score')
        if not self.kw['data.format_prompt']:
            self.kw['data.format_prompt'] = f'{self._folder_path}/easy_r1/format_prompt/math.jinja'

        cmd = f'python -m verl.trainer.main config={self._folder_path}/easy_r1/config.yaml '
        cmd += self.kw.parse_kwargs()
        cmd += f' 2>&1 | tee {self.log_file_path}'

        return cmd

    def __call__(self, *args, **kw):
        save_path = super().__call__(*args, **kw)
        ckpt_tracker_file = os.path.join(save_path, 'checkpoint_tracker.json')
        if not os.path.exists(ckpt_tracker_file):
            not_found_msg = 'Training failed, checkpoint_tracker.json not found.'
            LOG.error(not_found_msg)
            return not_found_msg
        with open(ckpt_tracker_file, 'r') as f:
            json_data = json.load(f)
        actor_path = json_data.get('last_actor_path', None)
        if not actor_path or not os.path.exists(actor_path):
            not_found_msg = 'Training failed, last_actor_path not found in checkpoint_tracker.json.'
            LOG.error(not_found_msg)
            return not_found_msg

        self._merge_ckpt(actor_path)
        huggingface_path = os.path.join(actor_path, 'huggingface')
        return huggingface_path

    def _merge_ckpt(self, path):
        try:
            script_path = f'{self._folder_path}/easy_r1/model_merger.py'
            subprocess.run(
                ['python', script_path, '--local_dir', path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except CalledProcessError as e:
            LOG.error(f'Command failed with return code {e.returncode}: {e.stderr}')
            return False
        except Exception as e:
            LOG.error(f'Error merging checkpoints: {e}')
            return False
        return True

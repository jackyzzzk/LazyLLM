import os
import copy
import random

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase


class AlpacaloraFinetune(LazyLLMFinetuneBase):
    """此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [alpaca-lora](https://github.com/tloen/alpaca-lora) 项目提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。

Args:
    base_model (str): 用于微调的基模型本地路径。
    target_path (str): 微调后 LoRA 权重保存路径。
    merge_path (Optional[str]): 合并 LoRA 权重后的模型保存路径，默认 ``None``。
        若未提供，则在 ``target_path`` 下创建 "lazyllm_lora" 与 "lazyllm_merge" 目录。
    model_name (Optional[str]): 模型名称，用于日志前缀，默认 ``LLM``。
    cp_files (Optional[str]): 从基模型路径复制配置文件到 ``merge_path``，默认 ``tokeniz*``。
    launcher (lazyllm.launcher): 微调启动器，默认 ``launchers.remote(ngpus=1)``。
    kw (dict): 用于更新默认训练参数的关键字参数，允许更新如下参数：

Keyword Args:
    data_path (Optional[str]): 数据路径，默认 ``None``。
    batch_size (Optional[int]): 批大小，默认 64。
    micro_batch_size (Optional[int]): 微批大小，默认 4。
    num_epochs (Optional[int]): 训练轮数，默认 2。
    learning_rate (Optional[float]): 学习率，默认 5.e-4。
    cutoff_len (Optional[int]): 截断长度，默认 1030。
    filter_nums (Optional[int]): 过滤器数量，默认 1024。
    val_set_size (Optional[int]): 验证集大小，默认 200。
    lora_r (Optional[int]): LoRA 秩，默认 8。
    lora_alpha (Optional[int]): LoRA 融合因子，默认 32。
    lora_dropout (Optional[float]): LoRA 丢弃率，默认 0.05。
    lora_target_modules (Optional[str]): LoRA 目标模块，默认 ``[wo,wqkv]``。
    modules_to_save (Optional[str]): 全量微调模块，默认 ``[tok_embeddings,output]``。
    deepspeed (Optional[str]): DeepSpeed 配置路径，默认使用仓库预制 ds.json。
    prompt_template_name (Optional[str]): 提示模板名称，默认 ``alpaca``。
    train_on_inputs (Optional[bool]): 是否在输入上训练，默认 ``True``。
    show_prompt (Optional[bool]): 是否显示提示，默认 ``False``。
    nccl_port (Optional[int]): NCCL 端口，默认随机在 19000-20500。


Examples:
    >>> from lazyllm import finetune
    >>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
    """
    defatult_kw = ArgsDict({
        'data_path': None,
        'batch_size': 64,
        'micro_batch_size': 4,
        'num_epochs': 2,
        'learning_rate': 5.e-4,
        'cutoff_len': 1030,
        'filter_nums': 1024,
        'val_set_size': 200,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_target_modules': '[wo,wqkv]',
        'modules_to_save': '[tok_embeddings,output]',
        'deepspeed': '',
        'prompt_template_name': 'alpaca',
        'train_on_inputs': True,
        'show_prompt': False,
        'nccl_port': 19081,
    })
    auto_map = {'micro_batch_size': 'micro_batch_size'}

    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 model_name='LLM',
                 cp_files='tokeniz*',
                 launcher=launchers.remote(ngpus=1),  # noqa B008
                 **kw
                 ):
        if not merge_path:
            save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
            target_path, merge_path = os.path.join(save_path, 'lazyllm_lora'), os.path.join(save_path, 'lazyllm_merge')
            os.makedirs(target_path, exist_ok=True)
            os.makedirs(merge_path, exist_ok=True)
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self.folder_path = os.path.dirname(os.path.abspath(__file__))
        deepspeed_config_path = os.path.join(self.folder_path, 'alpaca-lora', 'ds.json')
        self.kw = copy.deepcopy(self.defatult_kw)
        self.kw['deepspeed'] = deepspeed_config_path
        self.kw['nccl_port'] = random.randint(19000, 20500)
        self.kw.check_and_update(kw)
        self.merge_path = merge_path
        self.cp_files = cp_files
        self.model_name = model_name

    def cmd(self, trainset, valset=None) -> str:
        """生成用于执行Alpaca-LoRA微调和模型合并的shell命令序列。

Args:
    trainset (str): 训练数据集路径，支持相对data_path配置的路径或绝对路径
    valset (str, optional): 验证数据集路径，未指定时将从训练集中自动划分

**Returns:**

- str or list: 当不需要合并模型时返回单个命令字符串，需要合并时返回包含微调命令、合并命令和文件拷贝命令的列表


Examples:
    >>> from lazyllm import finetune
    >>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
    >>> cmd = trainer.cmd("my_dataset.json")
    """
        thirdparty.check_packages(['datasets', 'deepspeed', 'fire', 'numpy', 'peft', 'torch', 'transformers'])
        if not os.path.exists(trainset):
            defatult_path = os.path.join(lazyllm.config['data_path'], trainset)
            if os.path.exists(defatult_path):
                trainset = defatult_path
        if not self.kw['data_path']:
            self.kw['data_path'] = trainset

        run_file_path = os.path.join(self.folder_path, 'alpaca-lora', 'finetune.py')
        cmd = (f'python {run_file_path} '
               f'--base_model={self.base_model} '
               f'--output_dir={self.target_path} '
            )
        cmd += self.kw.parse_kwargs()
        cmd += f' 2>&1 | tee {os.path.join(self.target_path, self.model_name)}_$(date +"%Y-%m-%d_%H-%M-%S").log'

        if self.merge_path:
            run_file_path = os.path.join(self.folder_path, 'alpaca-lora', 'utils', 'merge_weights.py')

            cmd = [cmd,
                   f'python {run_file_path} '
                   f'--base={self.base_model} '
                   f'--adapter={self.target_path} '
                   f'--save_path={self.merge_path} ',
                   f' cp {os.path.join(self.base_model, self.cp_files)} {self.merge_path} '
                ]

        # cmd = 'realpath .'
        return cmd

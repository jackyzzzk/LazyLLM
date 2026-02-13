import os
import yaml
import json
import uuid
import tempfile
import random
from datetime import datetime

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase
from .llama_factory.model_mapping import match_longest_prefix, llamafactory_mapping_dict


class LlamafactoryFinetune(LazyLLMFinetuneBase):
    """此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。

Args:
    base_model: 用于进行训练的基模型路径。支持本地路径，若路径不存在则尝试从配置的模型路径中查找。
    target_path: 训练完成后，模型权重保存的目标路径。
    merge_path (str, optional): 模型合并LoRA权重后的保存路径，默认为None。
        如果未指定，将在 ``target_path`` 下自动创建两个目录：
        - "lazyllm_lora"（用于存放LoRA训练权重）
        - "lazyllm_merge"（用于存放合并后的模型权重）
    config_path (str, optional): 训练配置的 YAML 文件路径，默认None。
        如果未指定，则使用默认配置文件 ``llama_factory/sft.yaml``。
        配置文件支持覆盖默认训练参数。
    export_config_path (str, optional): LoRA权重合并导出配置的 YAML 文件路径，默认None。
        如果未指定，则使用默认配置文件 ``llama_factory/lora_export.yaml``。
    lora_r (int, optional): LoRA的秩（rank），若提供则覆盖配置中的 ``lora_rank``。
    modules_to_save (str, optional): 额外需要保存的模型模块名称列表，格式类似于Python列表字符串，如 "[module1,module2]"。
    lora_target_modules (str, optional): 目标LoRA微调的模块名称列表，格式同上。
    launcher (lazyllm.launcher, optional): 微调任务的启动器，默认为单卡同步远程启动器 ``launchers.remote(ngpus=1, sync=True)``。
    **kw: 关键字参数，用于动态覆盖默认训练配置中的参数。

此类的关键字参数及其默认值如下：

Keyword Args:
    stage (typing.Literal['pt', 'sft', 'rm', 'ppo', 'dpo', 'kto']): 默认值是：``sft``。将在训练中执行的阶段。
    do_train (bool): 默认值是：``True``。是否运行训练。
    finetuning_type (typing.Literal['lora', 'freeze', 'full']): 默认值是：``lora``。要使用的微调方法。
    lora_target (str): 默认值是：``all``。要应用LoRA的目标模块的名称。使用逗号分隔多个模块。使用`all`指定所有线性模块。
    template (typing.Optional[str]): 默认值是：``None``。用于构建训练和推理提示的模板。
    cutoff_len (int): 默认值是：``1024``。数据集中token化后输入的截止长度。
    max_samples (typing.Optional[int]): 默认值是：``1000``。出于调试目的，截断每个数据集的示例数量。
    overwrite_cache (bool): 默认值是：``True``。覆盖缓存的训练和评估集。
    preprocessing_num_workers (typing.Optional[int]): 默认值是：``16``。用于预处理的进程数。
    dataset_dir (str): 默认值是：``lazyllm_temp_dir``。包含数据集的文件夹的路径。如果没有明确指定，LazyLLM将在当前工作目录的 ``.temp`` 文件夹中生成一个 ``dataset_info.json`` 文件，供LLaMA-Factory使用。
    logging_steps (float): 默认值是：``10``。每X个更新步骤记录一次日志。应该是整数或范围在 ``[0,1)`` 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    save_steps (float): 默认值是：``500``。每X个更新步骤保存一次检查点。应该是整数或范围在 ``[0,1)`` 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    plot_loss (bool): 默认值是：``True``。是否保存训练损失曲线。
    overwrite_output_dir (bool): 默认值是：``True``。覆盖输出目录的内容。
    per_device_train_batch_size (int): 默认值是：``1``。每个GPU/TPU/MPS/NPU核心/CPU的训练批次的大小。
    gradient_accumulation_steps (int): 默认值是：``8``。在执行反向传播及参数更新前，要累积的更新步骤数。
    learning_rate (float): 默认值是：``1e-04``。AdamW的初始学习率。
    num_train_epochs (float): 默认值是：``3.0``。要执行的总训练周期数。
    lr_scheduler_type (typing.Union[transformers.trainer_utils.SchedulerType, str]): 默认值是：``cosine``。要使用的调度器类型。
    warmup_ratio (float): 默认值是：``0.1``。在总步骤的 ``warmup_ratio`` 分之一阶段内进行线性预热。
    fp16 (bool): 默认值是：``True``。是否使用fp16（混合）精度，而不是32位。
    ddp_timeout (typing.Optional[int]): 默认值是：``180000000``。覆盖分布式训练的默认超时时间（值应以秒为单位给出）。
    report_to (typing.Union[NoneType, str, typing.List[str]]): 默认值是：``tensorboard``。要将结果和日志报告到的集成列表。
    val_size (float): 默认值是：``0.1``。验证集的大小，应该是整数或范围在`[0,1)`的浮点数。
    per_device_eval_batch_size (int): 默认值是：``1``。每个GPU/TPU/MPS/NPU核心/CPU的验证集批次大小。
    eval_strategy (typing.Union[transformers.trainer_utils.IntervalStrategy, str]): 默认值是：``steps``。要使用的验证评估策略。
    eval_steps (typing.Optional[float]): 默认值是：``500``。每X个步骤运行一次验证评估。应该是整数或范围在`[0,1)`的浮点数。如果小于1，将被解释为总训练步骤的比例。


Examples:
    >>> from lazyllm import finetune
    >>> trainer = finetune.llamafactory('internlm2-chat-7b', 'path/to/target')
    <lazyllm.llm.finetune type=LlamafactoryFinetune>
    """
    auto_map = {
        'gradient_step': 'gradient_accumulation_steps',
        'micro_batch_size': 'per_device_train_batch_size',
    }

    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 config_path=None,
                 export_config_path=None,
                 lora_r=None,
                 modules_to_save=None,
                 lora_target_modules=None,
                 launcher=launchers.remote(ngpus=1, sync=True),  # noqa B008
                 **kw
                 ):
        if not os.path.exists(base_model):
            default_path = os.path.join(lazyllm.config['model_path'], base_model)
            if os.path.exists(default_path):
                base_model = default_path
        if not merge_path:
            normalized_target = os.path.normpath(target_path)
            if normalized_target.endswith('lazyllm_lora'):
                merge_path = normalized_target.replace('lazyllm_lora', 'lazyllm_merge')
            else:
                save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
                target_path = os.path.join(save_path, 'lazyllm_lora')
                merge_path = os.path.join(save_path, 'lazyllm_merge')
            os.makedirs(target_path, exist_ok=True)
            os.makedirs(merge_path, exist_ok=True)
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self.merge_path = merge_path
        self.temp_yaml_file = None
        self.temp_export_yaml_file = None
        self.config_path = config_path
        self.export_config_path = export_config_path
        self.config_folder_path = os.path.dirname(os.path.abspath(__file__))

        default_config_path = os.path.join(self.config_folder_path, 'llama_factory', 'sft.yaml')
        self.template_dict = ArgsDict(self._load_yaml(default_config_path))

        if self.config_path:
            self.template_dict.update(self._load_yaml(self.config_path))

        if lora_r:
            self.template_dict['lora_rank'] = lora_r
        if modules_to_save:
            self.template_dict['additional_target'] = modules_to_save.strip('[]')
        if lora_target_modules:
            self.template_dict['lora_target'] = lora_target_modules.strip('[]')
        self.template_dict['model_name_or_path'] = base_model
        self.template_dict['output_dir'] = target_path
        self.template_dict['template'] = self._get_template_name(base_model)

        # Filter kw to only include keys that exist in template_dict
        # This ensures check_and_update won't fail due to unexpected keys
        # Keys not in template_dict will be silently ignored (they may be used elsewhere)
        ignored_keys = set(kw.keys()) - set(self.template_dict.keys())
        if ignored_keys:
            lazyllm.LOG.info(f'Ignored parameters not in template_dict: {sorted(ignored_keys)}')
        filtered_kw = {k: v for k, v in kw.items() if k in self.template_dict}

        self.template_dict.check_and_update(filtered_kw)

        default_export_config_path = os.path.join(self.config_folder_path, 'llama_factory', 'lora_export.yaml')
        self.export_dict = ArgsDict(self._load_yaml(default_export_config_path))

        if self.export_config_path:
            self.export_dict.update(self._load_yaml(self.export_config_path))

        self.export_dict['model_name_or_path'] = base_model
        self.export_dict['adapter_name_or_path'] = target_path
        self.export_dict['export_dir'] = merge_path
        self.export_dict['template'] = self.template_dict['template']

        self.temp_folder = os.path.join(lazyllm.config['temp_dir'], 'llamafactory_config', str(uuid.uuid4())[:10])
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        self.log_file_path = None

    def _get_template_name(self, base_model):
        base_name = os.path.basename(base_model).lower()
        key_value = match_longest_prefix(base_name)
        if key_value:
            return key_value
        else:
            raise RuntimeError(f'Cannot find prfix of base_model({base_model}) '
                               f'in DEFAULT_TEMPLATE of LLaMA_Factory: {llamafactory_mapping_dict}')

    def _load_yaml(self, config_path):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict

    def _build_temp_yaml(self, updated_template_str, prefix='train_'):
        fd, temp_yaml_file = tempfile.mkstemp(prefix=prefix, suffix='.yaml', dir=self.temp_folder)
        with os.fdopen(fd, 'w') as temp_file:
            temp_file.write(updated_template_str)
        return temp_yaml_file

    def _build_temp_dataset_info(self, datapaths, stage=None):  # noqa C901
        if isinstance(datapaths, str):
            datapaths = [datapaths]
        elif isinstance(datapaths, list) and all(isinstance(item, str) for item in datapaths):
            pass
        else:
            raise TypeError(f'datapaths({datapaths}) should be str or list of str.')

        if stage is None:
            stage = self.template_dict.get('stage', 'sft').lower()

        supported_stages = ['sft', 'pt', 'dpo']
        if stage not in supported_stages:
            raise ValueError(
                f'Unsupported training stage: {stage}. '
                f'Only supported stages are: {", ".join(supported_stages)}'
            )

        temp_dataset_dict = dict()
        for datapath in datapaths:
            datapath = os.path.join(lazyllm.config['data_path'], datapath)
            assert os.path.isfile(datapath)
            file_name, _ = os.path.splitext(os.path.basename(datapath))
            temp_dataset_dict[file_name] = {'file_name': datapath}

            formatting = None
            first_item = None

            if stage == 'pt':
                formatting = None
                try:
                    with open(datapath, 'r', encoding='utf-8') as file:
                        first_bytes = file.read(1024)
                        file.seek(0)

                        if not first_bytes.strip().startswith(('[', '{')):
                            lines = file.readlines()
                            if not lines:
                                raise ValueError(f'PT stage: Dataset file {datapath} is empty')

                            first_item = {'text': lines[0].strip() if lines else ''}
                            lazyllm.LOG.info(
                                f'PT stage: Dataset {file_name} detected as plain text format '
                                f'({len(lines)} lines). LLaMA-Factory will handle conversion.'
                            )
                        else:
                            try:
                                data = json.load(file)
                                if isinstance(data, list):
                                    if not data:
                                        raise ValueError(f'PT stage: Dataset file {datapath} is empty (empty list)')
                                    first_item = data[0]
                                elif isinstance(data, dict):
                                    first_item = data
                                else:
                                    raise ValueError(
                                        f'PT stage: Dataset file {datapath} has invalid JSON structure. '
                                        f'Expected list or dict, got {type(data).__name__}'
                                    )
                                lazyllm.LOG.info(
                                    f'PT stage: Dataset {file_name} detected as JSON format. '
                                    f'Looking for "text" field.'
                                )
                            except json.JSONDecodeError as json_err:
                                raise ValueError(
                                    f'PT stage: Dataset file {datapath} is neither valid plain text nor valid JSON. '
                                    f'JSON parse error: {str(json_err)}'
                                )

                    if not first_item:
                        raise ValueError(f'PT stage: Failed to extract first item from dataset {datapath}')

                    self._build_alpaca_dataset_info(
                        temp_dataset_dict, file_name, first_item, stage
                    )

                except Exception as e:
                    error_msg = (
                        f'PT stage: Failed to process dataset {file_name} from {datapath}. '
                        f'Error: {str(e)}. '
                        f'PT mode requires either: '
                        f'(1) Plain text format (one text per line), or '
                        f'(2) JSON format with "text" field in each object.'
                    )
                    lazyllm.LOG.error(error_msg)
                    raise ValueError(error_msg) from e
            else:
                formatting = 'alpaca'
                try:
                    with open(datapath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    if not data:
                        raise ValueError(f'Dataset file {datapath} is empty')

                    first_item = data[0]

                    if 'messages' in first_item:
                        formatting = 'sharegpt'
                        self._build_sharegpt_dataset_info(
                            temp_dataset_dict, file_name, first_item, stage
                        )
                    else:
                        self._build_alpaca_dataset_info(
                            temp_dataset_dict, file_name, first_item, stage
                        )

                except Exception as e:
                    lazyllm.LOG.warning(
                        f'Failed to analyze dataset {datapath} for stage {stage}: {e}. '
                        f'Using default formatting.'
                    )

            if formatting is not None:
                temp_dataset_dict[file_name].update({'formatting': formatting})

        self.temp_dataset_info_path = os.path.join(self.temp_folder, 'dataset_info.json')
        with open(self.temp_dataset_info_path, 'w') as json_file:
            json.dump(temp_dataset_dict, json_file, indent=4)
        return self.temp_dataset_info_path, ','.join(temp_dataset_dict.keys())

    def _build_alpaca_dataset_info(self, dataset_dict, file_name, first_item, stage):  # noqa C901
        columns = {}
        ranking = False

        media_types = []
        for media in ['images', 'videos', 'audios']:
            if media in first_item:
                media_types.append(media)

        if stage == 'pt':
            if 'text' in first_item:
                columns['prompt'] = 'text'
            else:
                if 'instruction' in first_item:
                    columns['prompt'] = 'instruction'
                elif 'output' in first_item:
                    columns['prompt'] = 'output'
                else:
                    lazyllm.LOG.warning(
                        f'PT stage: No "text" field found in dataset {file_name}, '
                        f'using "instruction" or "output" as fallback'
                    )
                    columns['prompt'] = 'instruction' if 'instruction' in first_item else 'output'

        elif stage == 'dpo':
            ranking = True
            if 'chosen' in first_item and 'rejected' in first_item:
                columns['prompt'] = 'instruction' if 'instruction' in first_item else None
                columns['query'] = 'input' if 'input' in first_item else None
                columns['chosen'] = 'chosen'
                columns['rejected'] = 'rejected'
                columns = {k: v for k, v in columns.items() if v is not None}
            else:
                raise ValueError(
                    f'DPO stage requires "chosen" and "rejected" fields in dataset, '
                    f'but found: {list(first_item.keys())}'
                )

        elif stage == 'sft':
            if 'instruction' in first_item:
                columns['prompt'] = 'instruction'
            if 'input' in first_item:
                columns['query'] = 'input'
            if 'output' in first_item:
                columns['response'] = 'output'
            if 'system' in first_item:
                columns['system'] = 'system'
            if 'history' in first_item:
                columns['history'] = 'history'
        else:
            raise ValueError(f'Unsupported stage: {stage}. Only sft, pt, dpo are supported.')

        if media_types:
            multimodal_columns = {item: item for item in media_types}
            multimodal_columns.update(columns)
            columns = multimodal_columns

        update_dict = {'columns': columns}
        if ranking:
            update_dict['ranking'] = True

        dataset_dict[file_name].update(update_dict)

    def _build_sharegpt_dataset_info(self, dataset_dict, file_name, first_item, stage):  # noqa C901
        columns = {}
        ranking = False

        media_types = []
        for media in ['images', 'videos', 'audios']:
            if media in first_item:
                media_types.append(media)

        if stage == 'dpo':
            ranking = True
            if 'chosen' in first_item and 'rejected' in first_item:
                columns['messages'] = 'conversations' if 'conversations' in first_item else 'messages'
                columns['chosen'] = 'chosen'
                columns['rejected'] = 'rejected'
            else:
                raise ValueError(
                    f'DPO stage requires "chosen" and "rejected" fields in dataset, '
                    f'but found: {list(first_item.keys())}'
                )
        elif stage == 'sft':
            columns['messages'] = 'conversations' if 'conversations' in first_item else 'messages'
            if 'system' in first_item:
                columns['system'] = 'system'
            if 'tools' in first_item:
                columns['tools'] = 'tools'

            if 'messages' in first_item and isinstance(first_item['messages'], list):
                if len(first_item['messages']) > 0:
                    msg = first_item['messages'][0]
                    if 'role' in msg and 'content' in msg:
                        dataset_dict[file_name].update({
                            'tags': {
                                'role_tag': 'role',
                                'content_tag': 'content',
                                'user_tag': 'user',
                                'assistant_tag': 'assistant',
                                'system_tag': 'system'
                            }
                        })
        else:
            raise ValueError(f'Unsupported stage: {stage}. Only sft, pt, dpo are supported.')

        if media_types:
            multimodal_columns = {item: item for item in media_types}
            multimodal_columns.update(columns)
            columns = multimodal_columns

        update_dict = {'columns': columns}
        if ranking:
            update_dict['ranking'] = True

        dataset_dict[file_name].update(update_dict)

    def _rm_temp_yaml(self):
        if self.temp_yaml_file:
            if os.path.exists(self.temp_yaml_file):
                os.remove(self.temp_yaml_file)
            self.temp_yaml_file = None

    def cmd(self, trainset, valset=None) -> str:
        """生成LLaMA-Factory微调命令序列，包括训练和模型合并命令。

Args:
    trainset (str): 训练数据集路径(支持相对lazyllm.config['data_path']的路径)
    valset (str, optional): 验证数据集路径(当前实现中未直接使用)

**Returns:**

- str: 完整的shell命令字符串，包含:
    - 训练命令(自动配置参数)
    - 日志重定向(保存到目标路径)
    - 可选的模型合并命令(当配置LoRA时)

注意事项:
    - 自动生成带时间戳的训练日志文件
    - 临时文件会在使用后自动清理
    - 支持多种数据格式(alpaca/sharegpt等)
    - 多模态数据(图像/视频/音频)会自动检测处理
"""
        thirdparty.check_packages(['datasets', 'deepspeed', 'numpy', 'peft', 'torch', 'transformers', 'trl'])
        if 'dataset_dir' in self.template_dict and self.template_dict['dataset_dir'] == 'lazyllm_temp_dir':
            stage = self.template_dict.get('stage', 'sft')
            _, datasets = self._build_temp_dataset_info(trainset, stage=stage)
            self.template_dict['dataset_dir'] = self.temp_folder
        else:
            datasets = trainset
        self.template_dict['dataset'] = datasets

        if self.template_dict['finetuning_type'] == 'lora':
            # For LoRA/QLoRA: use llamafactory-cli export to merge adapter with base model
            # For Full finetuning: model copy handling is done in cmds below
            updated_export_str = yaml.dump(dict(self.export_dict), default_flow_style=False)
            self.temp_export_yaml_file = self._build_temp_yaml(updated_export_str, prefix='merge_')

        updated_template_str = yaml.dump(dict(self.template_dict), default_flow_style=False)
        self.temp_yaml_file = self._build_temp_yaml(updated_template_str)

        formatted_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_value = random.randint(1000, 9999)
        self.log_file_path = f'{self.target_path}/train_log_{formatted_date}_{random_value}.log'

        # Use bash instead of sh to support pipefail
        # This ensures export only runs if training succeeds
        cmds = (f'export DISABLE_VERSION_CHECK=1 && bash -c "set -o pipefail && '
                f'llamafactory-cli train {self.temp_yaml_file} 2>&1 | '
                f'tee {self.log_file_path}"')
        if self.temp_export_yaml_file:
            # For LoRA/QLoRA: merge adapter with base model
            # Only run export if training succeeds (exit code 0)
            # With pipefail, tee will preserve the exit code from llamafactory-cli
            cmds += f' && llamafactory-cli export {self.temp_export_yaml_file}'
        elif self.template_dict['finetuning_type'] == 'full':
            # For Full finetuning: copy model from lazyllm_lora to lazyllm_merge
            # This maintains consistency with LoRA/QLoRA workflow
            # Only copy if training succeeds (exit code 0)
            # Only copy model files, exclude training process information (checkpoints, logs, etc.)
            # to save storage space since lazyllm_merge is only used for exporting to models directory
            exclude_patterns = [
                '--exclude=checkpoint-*',  # Exclude checkpoint directories
                '--exclude=train_log_*.log',  # Exclude training logs
                '--exclude=trainer_state.json',  # Exclude trainer state
                '--exclude=trainer_log.jsonl',  # Exclude trainer log
                '--exclude=train_results.json',  # Exclude training results
                '--exclude=all_results.json',  # Exclude all results
                '--exclude=eval_results.json',  # Exclude evaluation results
                '--exclude=training_loss.png',  # Exclude training loss plot
                '--exclude=runs/',  # Exclude tensorboard logs
                '--exclude=training_args.bin',  # Exclude training arguments
            ]
            exclude_str = ' '.join(exclude_patterns)
            cmds += f' && rsync -a {exclude_str} {self.target_path}/ {self.merge_path}/ 2>/dev/null || true'
        return cmds

from typing import Any, Optional, Union
import lazyllm
from lazyllm import LOG
from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from .utils import get_candidate_entries, process_trainable_args, process_online_args


class AutoModel:
    """用于快速创建在线推理模块 OnlineModule 或本地 TrainableModule 的工厂类。它会优先采用用户传入的参数，若开启 ``config`` 则会根据 ``auto_model_config_map`` 中的配置进行覆盖，然后自动判断应当构建在线模块还是本地模块：

- 当判定为在线模块时，参数会透传给 OnlineModule（自动匹配 OnlineChatModule / OnlineEmbeddingModule / OnlineMultiModalModule）。

- 当判定为本地模块时，则以 ``model`` 与用户参数初始化 TrainableModule，并读取 config map 里的配置参数。

Args:
    model (str): 指定模型名称。例如 ``Qwen3-32B``。必填。
    config_id (Optional[str]): 指定配置文件里的id。默认为空。
    source (Optional[str]): 使用的服务提供方。为在线模块（``OnlineModule``）指定 ``qwen`` / ``glm`` / ``openai`` 等；若设为 ``local`` 则强制创建本地 TrainableModule。
    type (Optional[str]): 模型类型。若未指定会尝试从 kwargs 中获取或由在线模块自动推断。
    config (Union[str, bool]): 是否启用 ``auto_model_config_map`` 的覆盖逻辑，或者用户指定的 config 文件路径。默认为 True。
    **kwargs: 兼容 `model` 的同义字段 `base_model` 和 `embed_model_name`，不接收其他用户传入的字段。
"""

    def __new__(cls, model: Optional[str] = None, *, config_id: Optional[str] = None, source: Optional[str] = None,  # noqa C901
                type: Optional[str] = None, config: Union[str, bool] = True, **kwargs: Any):
        # check and accomodate user params
        model = model or kwargs.pop('base_model', kwargs.pop('embed_model_name', None))
        if model in lazyllm.online.chat:
            source, model = model, None

        if not model:
            try:
                return lazyllm.OnlineModule(source=source, type=type)
            except Exception as e:
                raise RuntimeError(f'`model` is not provided in AutoModel, and {e}') from None

        trainable_entry, online_entry = get_candidate_entries(model, config_id, source, config)

        # 1) first: try TrainableModule with trainable config (for directly connecting deployed endpoint)
        if trainable_entry is not None:
            trainable_args = process_trainable_args(
                model=model, type=type, source=source, config=config, entry=trainable_entry
            )
            try:
                module = TrainableModule(**trainable_args)
                if module._url or module._impl._get_deploy_tasks.flag: return module
            except Exception as e:
                LOG.warning('Fail to create `TrainableModule`, will try to '
                            f'load model {model} with `OnlineModule`. Since the error: {e}')

        # 2) second: try OnlineModule with online config if found
        if online_entry is not None:
            online_args = process_online_args(model=model, source=source, type=type, entry=online_entry)
            if online_args: return OnlineModule(**online_args)

        # 3) finally: fallback (no config or config unusable)
        try:
            return OnlineModule(model=model, source=source, type=type)
        except Exception as e:
            LOG.warning('`OnlineModule` creation failed, and will try to '
                        f'load model {model} with local `TrainableModule`. Since the error: {e}')
            return TrainableModule(model, type=type)

import os
import json
import lazyllm
from lazyllm.components.utils.file_operate import _base64_to_file, _is_base64_with_mime
from lazyllm import LOG, LazyLLMLaunchersBase
from lazyllm.thirdparty import transformers as tf, torch, sentence_transformers, numpy as np, FlagEmbedding as fe
from .base import LazyLLMDeployBase
from typing import Union, List, Dict, Optional
from abc import ABC, abstractmethod


class AbstractEmbedding(ABC):
    """抽象嵌入基类，为所有嵌入模型提供统一的接口和基础功能。此类定义了嵌入模型的标准接口，包括模型加载、调用和序列化等功能。

Args:
    base_embed (str): 嵌入模型的基础路径或标识符，用于指定要加载的嵌入模型。
    source (str, optional): 模型来源，默认为 ``None``。如果未指定，将使用 LazyLLM 配置中的默认模型来源。
    init (bool): 是否在初始化时立即加载模型，默认为 ``False``。如果为 ``True``，将在对象创建时立即调用 ``load_embed()`` 方法。
"""
    def __init__(self, base_embed, source=None, init=False):
        from ..utils.downloader import ModelManager
        self._source = source or lazyllm.config['model_source']
        self._base_embed = ModelManager(self._source).download(base_embed) or ''
        self._embed = None
        self._init = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self._init, self.load_embed)

    @abstractmethod
    def load_embed(self) -> None:
        """加载嵌入模型的抽象方法。此方法由子类实现，用于执行具体的模型加载逻辑。
"""
        pass

    @abstractmethod
    def _call(self, data: Dict[str, Union[str, List[str]]]) -> str:
        pass

    def __call__(self, data: Dict[str, Union[str, List[str]]]) -> str:
        lazyllm.call_once(self._init, self.load_embed)
        return self._call(data)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self._init)
        return self.__class__, (self._base_embed, self._source, init)

class LazyHuggingFaceDefaultEmbedding(AbstractEmbedding):

    def load_embed(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tokenizer = tf.AutoTokenizer.from_pretrained(self._base_embed, trust_remote_code=True)
        self._embed = tf.AutoModel.from_pretrained(self._base_embed, trust_remote_code=True).to(self._device)
        self._embed.eval()

    def _call(self, data: Dict[str, Union[str, List[str]]]):
        string, _ = data['text'], data['images']
        encoded_input = self._tokenizer(string, padding=True, truncation=True, return_tensors='pt',
                                        max_length=512, add_special_tokens=True).to(self._device)
        with torch.no_grad():
            model_output = self._embed(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        res = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy().tolist()
        if type(string) is str:
            return json.dumps(res[0])
        else:
            return json.dumps(res)

class HuggingFaceEmbedding:
    """HuggingFace嵌入模型管理类，用于管理和注册不同的嵌入模型实现。

属性：
    _model_id_mapping (dict): 模型ID到具体实现类的映射字典。

Args:
    base_embed (str): 基础嵌入模型的路径或名称。
    source (Optional[str]): 模型来源，默认为None。
"""
    _model_id_mapping = {}

    @classmethod
    def get_emb_cls(cls, model_name: str):
        """获取模型对应的嵌入实现类。

Args:
    model_name (str): 模型名称或路径。

**Returns:**

- type: 返回对应的嵌入模型实现类，如果未找到则返回默认实现LazyHuggingFaceDefaultEmbedding。
"""
        model_id = model_name.split('/')[-1].lower()
        return cls._model_id_mapping.get(model_id, LazyHuggingFaceDefaultEmbedding)

    @classmethod
    def register(cls, model_ids: List[str]):
        """注册模型ID到特定实现类的装饰器。

Args:
    model_ids (List[str]): 要注册的模型ID列表。

**Returns:**

- Callable: 返回装饰器函数。
"""
        def decorator(target_class):
            for ele in model_ids:
                cls._model_id_mapping[ele.lower()] = target_class
            return target_class
        return decorator

    def __init__(self, base_embed, source=None):
        self._embed = self.__class__.get_emb_cls(base_embed)(base_embed, source)

    def load_embed(self):
        """加载嵌入模型。

该方法会调用内部嵌入实现类的load_embed方法来加载模型。
"""
        self._embed.load_embed()

    def __call__(self, *args, **kwargs):
        try:
            args[0]['images'] = [_base64_to_file(image) if _is_base64_with_mime(image) else image
                                 for image in args[0]['images']]
        except Exception as e:
            LOG.error(f'Error converting base64 to image: {e}')
        return self._embed(*args, **kwargs)

class LazyFlagEmbedding(object):
    """支持懒加载的 FlagEmbedding 嵌入模块封装。

该类包装了 FlagEmbedding 的加载和调用逻辑，提供对稀疏和稠密嵌入的支持，并通过 lazyllm.once_flag() 机制实现懒加载。适用于嵌入模型的本地/远程下载、初始化与编码流程的封装，便于与 LazyLLM 系统集成。

Args:
    base_embed (str): 嵌入模型名称或路径。
    sparse (bool): 是否使用稀疏嵌入模式，默认为 False。
    source (str, optional): 模型下载源，若未提供则使用 lazyllm 全局配置。
    init (bool): 是否在初始化时立即加载模型，默认为 False。
"""
    def __init__(self, base_embed, sparse=False, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_embed = ModelManager(source).download(base_embed) or ''
        self.embed = None
        self.device = 'cpu'
        self.sparse = sparse
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_embed)

    def load_embed(self):
        """加载嵌入模型并初始化到设备上。

该方法根据系统是否支持 CUDA 自动选择运行设备（GPU 或 CPU），并从本地或远程加载预训练的 FlagEmbedding 模型。
"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed = fe.FlagAutoModel.from_finetuned(self.base_embed, use_fp16=False, devices=[self.device])

    def __call__(self, data: Dict[str, Union[str, List[str]]]):
        lazyllm.call_once(self.init_flag, self.load_embed)
        string, _ = data['text'], data['images']
        with torch.no_grad():
            model_output = self.embed.encode(string, return_sparse=self.sparse)
        if self.sparse:
            embeddings = model_output['lexical_weights']
            if isinstance(string, list):
                res = [dict(embedding) for embedding in embeddings]
            else:
                res = dict(embeddings)
        else:
            res = model_output['dense_vecs'].tolist()

        if type(string) is list and type(res) is dict:
            return json.dumps([res], default=lambda x: float(x))
        else:
            return json.dumps(res, default=lambda x: float(x))

    @classmethod
    def rebuild(cls, base_embed, sparse, init):
        """重建 LazyFlagEmbedding 实例的方法。

该类方法用于在序列化或跨进程传递时，重新构造带有初始化配置的 LazyFlagEmbedding 实例。

Args:
    base_embed (str): 嵌入模型的路径或模型名称。
    sparse (bool): 是否启用稀疏嵌入。
    init (bool): 是否在构造时立即加载模型。

**Returns:**

- LazyFlagEmbedding: 一个新的 LazyFlagEmbedding 实例。
"""
        return cls(base_embed, sparse, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyFlagEmbedding.rebuild, (self.base_embed, self.sparse, init)

class LazyHuggingFaceRerank(object):
    """基于 HuggingFace CrossEncoder 的重排序（Rerank）封装类。  
用于根据查询与候选文档的相关性分数，对文档进行排序。  
支持在初始化时下载并加载指定的重排序模型，并可选择延迟加载以提升启动性能。

Args:
    base_rerank (str): 重排序模型名称或本地路径。支持 HuggingFace Hub 模型标识符或本地路径。
    source (Optional[str]): 模型来源，支持 `huggingface` 和 `modelscope`，默认为全局配置项 `model_source`。
    init (bool): 是否在实例化时立即加载模型。若为 `False`，将在首次调用时延迟加载。
"""
    def __init__(self, base_rerank, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_rerank = ModelManager(source).download(base_rerank) or ''
        self.reranker = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_reranker)

    def load_reranker(self):
        """加载重排序模型。  

该方法会基于 `self.base_rerank` 初始化一个 `sentence_transformers.CrossEncoder` 实例，  
并赋值给类属性 `self.reranker`，用于后续的重排序任务。  
"""
        self.reranker = sentence_transformers.CrossEncoder(self.base_rerank)

    def __call__(self, inps):
        lazyllm.call_once(self.init_flag, self.load_reranker)
        query, documents, top_n = inps['query'], inps['documents'], inps['top_n']
        query_pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(query_pairs)
        sorted_indices = [(index, scores[index]) for index in np.argsort(scores)[::-1]]
        if top_n > 0:
            sorted_indices = sorted_indices[:top_n]
        return sorted_indices

    @classmethod
    def rebuild(cls, base_rerank, init):
        """重建 `LazyHuggingFaceRerank` 实例的类方法。  
主要用于序列化（pickle/cloudpickle）时的反序列化过程，根据提供的参数重新实例化对象。

Args:
    base_rerank (str): 模型名称或路径。
    init (bool): 是否在重建时立即加载模型。

**Returns:**

- LazyHuggingFaceRerank: 重新构建的类实例。
"""
        return cls(base_rerank, init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyHuggingFaceRerank.rebuild, (self.base_rerank, init)

class EmbeddingDeploy(LazyLLMDeployBase):
    """此类是 ``LazyLLMDeployBase`` 的子类，用于部署文本嵌入（Embedding）服务。支持稠密向量（dense）和稀疏向量（sparse）两种嵌入方式，可使用 HuggingFace 模型或 FlagEmbedding 模型。

Args:
    launcher (Optional[lazyllm.launcher]): 启动器实例，默认为 ``None``。
    model_type (Optional[str]): 模型类型，默认为 ``'embed'``。
    log_path (Optional[str]): 日志文件路径，默认为 ``None``。
    embed_type (Optional[str]): 嵌入类型，可选 ``'dense'`` 或 ``'sparse'``，默认为 ``'dense'``。
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。
    port (Optional[int]): 服务端口号，默认为 ``None``，此情况下 LazyLLM 会自动生成随机端口号。

Call Arguments:
    finetuned_model (Optional[str]): 微调后的模型路径或名称。\n
    base_model (Optional[str]): 基础模型路径或名称，当 finetuned_model 无效时会使用此模型。\n

Message Format:
    输入格式为包含 text（文本）和 images（图像列表）的字典。\n
    - text (str): 需要编码的文本内容 \n
    - images (Union[str, List[str]]): 需要编码的图像列表，可选 \n


Examples:
    >>> from lazyllm import deploy
    >>> embed_service = deploy.EmbeddingDeploy(embed_type='dense')
    >>> embed_service('path/to/model')
    """
    message_format = {
        'text': 'text',  # str,
        'images': []  # Union[str, List[str]]
    }
    keys_name_handle = {
        'inputs': 'text',
        'image': 'images'
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher: LazyLLMLaunchersBase = None, model_type: str = 'embed', log_path: Optional[str] = None,
                 embed_type: Optional[str] = 'dense', trust_remote_code: bool = True, port: Optional[int] = None, **kw):
        super().__init__(launcher=launcher)
        self._launcher = launcher
        self._port = port
        self._model_type = model_type
        self._log_path = log_path
        self._sparse_embed = True if embed_type == 'sparse' else False
        self._trust_remote_code = trust_remote_code
        self._port = port

    def _get_model_path(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model
        return finetuned_model

    def __call__(self, finetuned_model=None, base_model=None):
        finetuned_model = self._get_model_path(finetuned_model, base_model)
        if self._sparse_embed or lazyllm.config['default_embedding_engine'] == 'flagEmbedding':
            return lazyllm.deploy.RelayServer(port=self._port, func=LazyFlagEmbedding(
                finetuned_model, sparse=self._sparse_embed),
                launcher=self._launcher, log_path=self._log_path, cls='embedding')()
        else:
            return lazyllm.deploy.RelayServer(port=self._port, func=HuggingFaceEmbedding(finetuned_model),
                                              launcher=self._launcher, log_path=self._log_path, cls='embedding')()


@HuggingFaceEmbedding.register(model_ids=['BGE-VL-v1.5-mmeb'])
class _BGEVLEmbedding(AbstractEmbedding):

    def __init__(self, base_embed, source=None, init=False):
        super().__init__(base_embed, source, init)

    def load_embed(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._embed = tf.AutoModel.from_pretrained(self._base_embed, trust_remote_code=True).to(self._device)
        self._embed.set_processor(self._base_embed)
        self._embed.processor.patch_size = self._embed.config.vision_config.patch_size
        self._embed.processor.vision_feature_select_strategy = self._embed.config.vision_feature_select_strategy
        self._embed.eval()

    def _call(self, data: Dict[str, Union[str, List[str]]]):
        DEFAULT_INSTRUCTION = 'Retrieve the target image that best meets the combined criteria by ' \
            'using both the provided image and the image retrieval instructions: '
        with torch.no_grad():
            # text='Make the background dark, as if the camera has taken the photo at night'
            # images='./cir_query.png'
            text, images = data['text'], data['images'][0] if isinstance(data['images'], list) else data['images']

            query_inputs = self._embed.data_process(
                text=text,
                images=images,
                q_or_c='q',
                task_instruction=DEFAULT_INSTRUCTION
            )
            query_embs = self._embed(**query_inputs, output_hidden_states=True)[:, -1, :]
            res = torch.nn.functional.normalize(query_embs, dim=-1).cpu().numpy().tolist()
            return json.dumps(res[0])


class RerankDeploy(EmbeddingDeploy):
    """此类是 ``EmbeddingDeploy`` 的子类，用于部署重排序（Rerank）服务。支持使用HuggingFace模型进行文本重排序。

Args:
    launcher (lazyllm.launcher): 启动器，默认为 ``None``。
    model_type (str): 模型类型，默认为 ``'embed'``。
    log_path (str): 日志文件路径，默认为 ``None``。
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。
    port (int): 服务端口号，默认为 ``None``，此情况下LazyLLM会自动生成随机端口号。

Call Arguments:
    finetuned_model: 微调后的模型路径或模型名称。

    base_model: 基础模型路径或模型名称，当finetuned_model无效时会使用此模型。


Message Format:
    输入格式为包含query（查询文本）、documents（候选文档列表）和top_n（返回的文档数量）的字典。

    - query: 查询文本

    - documents: 候选文档列表

    - top_n: 返回的文档数量，默认为1



Examples:
    >>> from lazyllm import deploy
    >>> rerank_service = deploy.embed.RerankDeploy()
    >>> rerank_service('path/to/model')
    >>> input_data = {
    ...     "query": "What is machine learning?",
    ...     "documents": [
    ...         "Machine learning is a branch of AI.",
    ...         "Machine learning uses data to improve.",
    ...         "Deep learning is a subset of machine learning."
    ...     ],
    ...     "top_n": 2
    ... }
    >>> result = rerank_service(input_data)
    """
    message_format = {'query': 'query', 'documents': ['string'], 'top_n': 1}
    keys_name_handle = {'inputs': 'query', 'documents': 'documents', 'top_n': 'top_n'}
    default_headers = {'Content-Type': 'application/json'}

    def __call__(self, finetuned_model=None, base_model=None):
        finetuned_model = self._get_model_path(finetuned_model, base_model)
        return lazyllm.deploy.RelayServer(port=self._port, func=LazyHuggingFaceRerank(
            finetuned_model), launcher=self._launcher, log_path=self._log_path, cls='embedding')()

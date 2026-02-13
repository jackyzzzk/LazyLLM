from typing import Any, Dict, Optional
from .onlinemodule import (
    OnlineChatModule, OnlineEmbeddingModule, OnlineMultiModalModule,
    OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
)
from lazyllm.components.utils.downloader.model_downloader import LLMType
from .onlinemodule.map_model_type import get_model_type


class _OnlineModuleMeta(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        return isinstance(__instance, (OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase))


class OnlineModule(metaclass=_OnlineModuleMeta):
    """在线模型基类，用来管理创建目前市面上公开的在线模型推理服务，包括LLM模块、Embedding模块以及多模态模块。
根据用户指定的在线模型类型和模型名自动创建对应的模块实例，目前支持的实例类型包括OnlineChatModule, OnlineEmbeddingModule和OnlineMultiModalModule。

Args:
    type (Optional[str]): 指定在线模型服务的类型，如果不指定则默认为 ``llm``。目前支持 ``llm`` / ``vlm`` / ``embed`` / ``cross_modal_embed`` / ``rerank`` / ``stt`` / ``tts`` / ``sd`` 这几类。
    model (Optional[str]): 指定要加载的模型名称，例如 ``internlm2-chat-7b``，可为空。为空时默认加载 ``internlm2-chat-7b``。
    source (Optional[str]): 指定要创建的模块类型，可选为 ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` 等。
    url (Optional[str]): 指定要访问的平台的基础链接，默认是官方链接
    **kwargs: 其他传递给基类的参数。


Examples:
    >>> import lazyllm
    >>> chat = lazyllm.OnlineModule(model="qwen-plus", source="qwen")
    >>> isinstance(chat, lazyllm.OnlineChatModule)
    True
    >>> print(chat("Say hi in one sentence."))
    Hi there! Happy to help.
    >>> embed = lazyllm.OnlineModule(type="embed", source="qwen", model="text-embedding-v1")
    >>> isinstance(embed, lazyllm.OnlineEmbeddingModule)
    True
    >>> vec = embed("LazyLLM routes models automatically.")
    >>> len(vec)
    1536
    >>> tts = lazyllm.OnlineModule(type="tts", source="qwen", model="qwen-tts")
    >>> isinstance(tts, lazyllm.OnlineMultiModalModule)
    True
    >>> audio_bytes = tts("Convert this line to speech.")
    >>> len(audio_bytes) > 0
    True
    """

    _EMBED_TYPES = (LLMType.EMBED, LLMType.CROSS_MODAL_EMBED, LLMType.RERANK)
    _MULTI_TYPE_TO_FUNCTION = {
        LLMType.STT: 'stt',
        LLMType.TTS: 'tts',
        LLMType.SD: 'text2image',
        LLMType.TEXT2IMAGE: 'text2image',
        LLMType.IMAGE_EDITING: 'image_editing',
    }

    def __new__(self, model: Optional[str] = None, source: Optional[str] = None, *,
                type: Optional[str] = None, url: Optional[str] = None, **kwargs):
        params: Dict[str, Any] = dict(kwargs)
        resolved_type = type or params.pop('function', None)
        if not resolved_type:
            resolved_type = (get_model_type(model) or 'llm') if model else 'llm'
        resolved_type = resolved_type.lower()

        if resolved_type in self._EMBED_TYPES:
            embed_kwargs = params.copy()
            embed_kwargs.pop('function', None)
            embed_kwargs.setdefault('type', 'rerank' if resolved_type == LLMType.RERANK else 'embed')
            return OnlineEmbeddingModule(source=source,
                                         embed_url=url,
                                         embed_model_name=model,
                                         **embed_kwargs)

        if resolved_type in list(self._MULTI_TYPE_TO_FUNCTION.keys()):
            multi_kwargs = params.copy()
            return OnlineMultiModalModule(model=model, source=source, base_url=url,
                                          type=self._MULTI_TYPE_TO_FUNCTION[LLMType(resolved_type)],
                                          **multi_kwargs)

        chat_kwargs = params.copy()
        chat_kwargs.pop('function', None)
        chat_kwargs.setdefault('type', resolved_type)
        return OnlineChatModule(model=model, source=source, base_url=url, **chat_kwargs)

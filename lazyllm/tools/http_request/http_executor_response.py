from lazyllm.thirdparty import httpx

class HttpExecutorResponse:
    """HTTP执行器响应类，用于封装和处理HTTP请求的响应结果。

提供对HTTP响应内容的统一访问接口，支持文件类型检测和内容提取。

Args:
    response (httpx.Response, optional): httpx库的响应对象，默认为None


**Returns:**

- HttpExecutorResponse实例，提供多种响应内容访问方式
"""
    headers: dict[str, str]
    response: 'httpx.Response'

    def __init__(self, response: 'httpx.Response' = None):
        self.response = response
        self.headers = dict(response.headers) if isinstance(self.response, httpx.Response) else {}

    @property
    def is_file(self) -> bool:
        content_type = self.get_content_type()
        file_content_types = ['image', 'audio', 'video']

        return any(v in content_type for v in file_content_types)

    def get_content_type(self) -> str:
        """获取HTTP响应的内容类型。

从响应头中提取 'content-type' 字段的值，用于判断响应内容的类型。

**Returns:**

- str: 响应的内容类型，如果未找到则返回空字符串。


Examples:
    >>> from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
    >>> import httpx
    >>> response = httpx.Response(200, headers={'content-type': 'application/json'})
    >>> http_response = HttpExecutorResponse(response)
    >>> content_type = http_response.get_content_type()
    >>> print(content_type)
    ... 'application/json'
    """
        return self.headers.get('content-type', '')

    def extract_file(self) -> tuple[str, bytes]:
        """从HTTP响应中提取文件内容。

如果响应内容类型是文件相关类型（如图片、音频、视频），则提取文件的内容类型和二进制数据。

**Returns:**

- tuple[str, bytes]: 包含内容类型和文件二进制数据的元组。如果不是文件类型，则返回空字符串和空字节。


Examples:
    >>> from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
    >>> import httpx
    >>> # 模拟图片响应
    >>> response = httpx.Response(200, headers={'content-type': 'image/jpeg'}, content=b'fake_image_data')
    >>> http_response = HttpExecutorResponse(response)
    >>> content_type, file_data = http_response.extract_file()
    >>> print(content_type)
    ... 'image/jpeg'
    >>> print(len(file_data))
    ... 15
    >>> # 模拟JSON响应
    >>> response = httpx.Response(200, headers={'content-type': 'application/json'}, content=b'{"key": "value"}')
    >>> http_response = HttpExecutorResponse(response)
    >>> content_type, file_data = http_response.extract_file()
    >>> print(content_type)
    ... ''
    >>> print(file_data)
    ... b''
    """
        if self.is_file:
            return self.get_content_type(), self.body

        return '', b''

    @property
    def content(self) -> str:
        if isinstance(self.response, httpx.Response):
            return self.response.text
        else:
            raise ValueError(f'Invalid response type {type(self.response)}')

    @property
    def body(self) -> bytes:
        if isinstance(self.response, httpx.Response):
            return self.response.content
        else:
            raise ValueError(f'Invalid response type {type(self.response)}')

    @property
    def status_code(self) -> int:
        if isinstance(self.response, httpx.Response):
            return self.response.status_code
        else:
            raise ValueError(f'Invalid response type {type(self.response)}')

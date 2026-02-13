import os
from pathlib import Path
from urllib.parse import urlparse

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register


if (
    'data' in LazyLLMRegisterMetaClass.all_clses
    and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']
):
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


def _is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def _is_pdf_url(url):
    try:
        import requests

        response = requests.head(
            url,
            allow_redirects=True,
            timeout=10,
        )
        return (
            response.status_code == 200
            and response.headers.get('Content-Type') == 'application/pdf'
        )
    except Exception:
        return False


def _download_pdf(url, save_path):
    try:
        import requests

        response = requests.get(
            url,
            stream=True,
            timeout=30,
        )

        if (
            response.status_code == 200
            and response.headers.get('Content-Type') == 'application/pdf'
        ):
            pdf_folder = os.path.dirname(save_path)
            os.makedirs(pdf_folder, exist_ok=True)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            LOG.info(f'PDF saved to {save_path}')
            return save_path

        LOG.warning('The URL did not return a valid PDF file.')
        return None

    except Exception as e:
        LOG.error(f'Error downloading PDF: {e}')
        return None


class FileOrURLNormalizer(kbc):
    """文件或URL标准化算子。

该算子根据输入类型（文件或URL）自动识别文件格式，进行标准化处理。
支持PDF、HTML/XML、TXT/MD等文件格式，以及网页URL。对于网络PDF，会先下载到本地。

Args:
    intermediate_dir (str): 中间文件保存目录，默认为 'intermediate'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 标准化后的数据，包含以下字段：
    - _type: 文件类型 ('pdf', 'html', 'text', 'invalid', 'unsupported')
    - _raw_path: 本地文件路径（如果有）
    - _url: URL地址（如果是网页）
    - _output_path: 预期的Markdown输出路径
    - _error: 错误信息（如果有）


Examples:
    ```python
    from lazyllm.tools.data import kbc

    normalizer = kbc.FileOrURLNormalizer(intermediate_dir='./temp')

    # For file input
    data = {'source': '/path/to/document.pdf'}
    result = normalizer(data)
    # Returns: {'source': '/path/to/document.pdf', '_type': 'pdf', '_raw_path': '/path/to/document.pdf', '_output_path': './temp/document.md'}

    # For URL input
    data = {'source': 'https://example.com/page.html'}
    result = normalizer(data)
    # Returns: {'source': 'https://example.com/page.html', '_type': 'html', '_url': 'https://example.com/page.html', '_output_path': './temp/url_xxx.md'}
    ```
    """
    def __init__(self, intermediate_dir: str = 'intermediate', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.intermediate_dir = intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)

    def forward(
        self,
        data: dict,
        input_key: str = 'source',
        **kwargs,
    ) -> dict:
        src = data.get(input_key, '')
        if not src:
            return {**data, '_type': 'invalid', '_error': 'Empty source'}

        result = data.copy()

        if _is_url(src):
            if _is_pdf_url(src):
                pdf_path = os.path.join(
                    self.intermediate_dir,
                    f'crawled_{id(data)}.pdf',
                )
                downloaded_path = _download_pdf(src, pdf_path)

                if downloaded_path:
                    result['_type'] = 'pdf'
                    result['_raw_path'] = downloaded_path
                else:
                    result['_type'] = 'invalid'
                    result['_error'] = 'Failed to download PDF from URL'
            else:
                result['_type'] = 'html'
                result['_url'] = src

        else:
            if not os.path.exists(src):
                result['_type'] = 'invalid'
                result['_error'] = f'File not found: {src}'
            else:
                ext = Path(src).suffix.lower()

                if ext in [
                    '.pdf',
                    '.png',
                    '.jpg',
                    '.jpeg',
                    '.webp',
                    '.gif',
                ]:
                    result['_type'] = 'pdf'
                    result['_raw_path'] = src

                elif ext in ['.html', '.xml']:
                    result['_type'] = 'html'
                    result['_raw_path'] = src

                elif ext in ['.txt', '.md']:
                    result['_type'] = 'text'
                    result['_raw_path'] = src

                else:
                    result['_type'] = 'unsupported'
                    result['_error'] = f'Unsupported file type: {ext}'

        if '_raw_path' in result:
            name = Path(result['_raw_path']).stem
            result['_output_path'] = os.path.join(
                self.intermediate_dir,
                f'{name}.md',
            )

        elif '_url' in result:
            result['_output_path'] = os.path.join(
                self.intermediate_dir,
                f'url_{id(data)}.md',
            )

        return result


class HTMLToMarkdownConverter(kbc):
    """HTML转Markdown转换器算子。

该算子使用trafilatura库从HTML或XML文件中提取内容并转换为Markdown格式。
支持本地HTML文件和网络URL，会自动处理页面元数据。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 转换后的数据，包含以下字段：
    - _markdown_path: 生成的Markdown文件路径


Examples:
    ```python
    from lazyllm.tools.data import kbc

    converter = kbc.HTMLToMarkdownConverter()

    # After normalization
    data = {'_type': 'html', '_url': 'https://example.com/article', '_output_path': './temp/output.md'}
    result = converter(data)
    # Returns: {'_type': 'html', '_url': 'https://example.com/article', '_output_path': './temp/output.md', '_markdown_path': './temp/output.md'}
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(self, data: dict, **kwargs) -> dict:
        if data.get('_type', '') != 'html':
            return data

        try:
            from trafilatura import fetch_url, extract
        except ImportError:
            LOG.error('trafilatura is not installed. Please install it with `pip install trafilatura`.')
            return {**data, '_markdown_path': ''}

        url = data.get('_url')
        raw_path = data.get('_raw_path')
        output_path = data.get('_output_path', '')

        try:
            if url:
                downloaded = fetch_url(url)
                if not downloaded:
                    error_msg = (
                        'fail to fetch this url. '
                        'Please check your Internet Connection or URL correctness'
                    )
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(error_msg)
                    return {**data, '_markdown_path': output_path}

            elif raw_path:
                with open(raw_path, 'r', encoding='utf-8') as f:
                    downloaded = f.read()
            else:
                return {**data, '_markdown_path': ''}

            result = extract(
                downloaded,
                output_format='markdown',
                with_metadata=True,
            )

            if result:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)

                LOG.info(f'Extracted content written to {output_path}')
                return {**data, '_markdown_path': output_path}

            return {**data, '_markdown_path': ''}

        except Exception as e:
            LOG.error(f'Error extracting HTML/XML: {e}')
            return {**data, '_markdown_path': ''}


class PDFToMarkdownConverterAPI(kbc):
    """PDF转Markdown转换器API算子。

该算子使用MinerU服务将PDF文件（包括扫描件和图片）转换为Markdown格式。
支持通过API调用MinerU进行PDF解析，可配置后端引擎和上传模式。

Args:
    mineru_url (str): MinerU服务URL地址。
    mineru_backend (str): MinerU后端引擎类型，默认为 'vlm-vllm-async-engine'。
    upload_mode (bool): 是否使用上传模式，默认为 True。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 转换后的数据，包含以下字段：
    - _markdown_path: 生成的Markdown文件路径


Examples:
    ```python
    from lazyllm.tools.data import kbc

    converter = kbc.PDFToMarkdownConverterAPI(
        mineru_url='your_mineru_url',
        mineru_backend='vlm-vllm-async-engine',
        upload_mode=True
    )

    # After normalization
    data = {'_type': 'pdf', '_raw_path': '/path/to/doc.pdf', '_output_path': './temp/output.md'}
    result = converter(data)
    # Returns: {'_type': 'pdf', '_raw_path': '/path/to/doc.pdf', '_output_path': './temp/output.md', '_markdown_path': './temp/output.md'}
    ```
    """
    def __init__(
        self,
        mineru_url: str = None,
        mineru_backend: str = 'vlm-vllm-async-engine',
        upload_mode: bool = True,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.mineru_url = mineru_url
        self.mineru_backend = mineru_backend
        self.upload_mode = upload_mode

    def forward(self, data: dict, **kwargs) -> dict:
        if data.get('_type', '') != 'pdf':
            return data

        if self.mineru_url is None:
            LOG.error('mineru_url is required for PDF processing')
            return {**data, '_markdown_path': ''}

        try:
            from lazyllm.tools.rag import MineruPDFReader
        except ImportError:
            LOG.error('MineruPDFReader not available')
            return {**data, '_markdown_path': ''}

        raw_path = data.get('_raw_path')
        output_path = data.get('_output_path', '')

        if not raw_path:
            return {**data, '_markdown_path': ''}

        try:
            reader = MineruPDFReader(
                url=self.mineru_url,
                backend=self.mineru_backend,
                upload_mode=self.upload_mode,
                split_doc=False,
            )

            docs = reader(file=raw_path, use_cache=False)

            if not docs:
                LOG.warning(f'MinerU returned no documents for: {raw_path}')
                return {**data, '_markdown_path': ''}

            md_content = '\n'.join(
                doc.text for doc in docs if doc.text
            )

            if not md_content.strip():
                LOG.warning(
                    f'MinerU returned empty content for: {raw_path}',
                )
                return {**data, '_markdown_path': ''}

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            LOG.info(f'MinerU parsed: {raw_path} -> {output_path}')
            return {**data, '_markdown_path': output_path}

        except Exception as e:
            LOG.error(f'MinerU API failed for {raw_path}: {e}')
            return {**data, '_markdown_path': ''}

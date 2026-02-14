"""Test cases for Knowledge Cleaning operators"""
import pytest
import json
import tempfile
import os
import shutil
from unittest.mock import Mock
import lazyllm
from lazyllm import config
from lazyllm.tools.data import kbc


class TestKnowledgeCleaningOperators:
    """Test suite for Knowledge Cleaning operators"""

    def setup_method(self):
        """Set up test environment"""
        self.root_dir = './test_kbc_data_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment"""
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def model_set(self):
        """Set up LLM model for testing"""
        # Use online chat model for testing
        self.llm = lazyllm.OnlineChatModule()
        return self.llm

    # ============ KBCChunkGenerator Tests ============
    
    def test_kbc_chunk_generator_init_token_method(self):
        """Test KBCChunkGenerator initialization with token method"""
        generator = kbc.KBCChunkGenerator(
            chunk_size=256,
            chunk_overlap=25,
            split_method="token",
            tokenizer_name="bert-base-uncased",
            _save_data=False
        )
        assert generator is not None
        assert generator.chunk_size == 256
        assert generator.chunk_overlap == 25
        assert generator.split_method == "token"

    def test_kbc_chunk_generator_forward(self):
        """Test KBCChunkGenerator forward processing"""
        # Create a test text file
        test_text = "This is a test sentence. " * 50  # Long enough to be split
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
        
        generator = kbc.KBCChunkGenerator(
            chunk_size=128,
            chunk_overlap=10,
            split_method="token",
            _save_data=False,
            _concurrency_mode='single'
        )
        
        # Test with single item (forward mode)
        input_data = [{'text_path': test_file}]
        results = generator(input_data)
        
        # Should return multiple chunks
        assert isinstance(results, list)
        assert len(results) > 1  # Text should be split into multiple chunks
        assert all('raw_chunk' in item for item in results)
        assert all(isinstance(item['raw_chunk'], str) for item in results)

    def test_kbc_chunk_generator_invalid_path(self):
        """Test KBCChunkGenerator with invalid file path"""
        generator = kbc.KBCChunkGenerator(
            _save_data=False,
            _concurrency_mode='single',
            _ignore_errors=True
        )
        
        input_data = [{'text_path': '/nonexistent/file.txt'}]
        results = generator(input_data)
        
        # Should handle error gracefully and return empty or error record
        assert isinstance(results, list)

    # ============ KBCChunkGeneratorBatch Tests ============

    def test_kbc_chunk_generator_batch_init(self):
        """Test KBCChunkGeneratorBatch initialization"""
        generator = kbc.KBCChunkGeneratorBatch(
            chunk_size=256,
            split_method="token",
            _save_data=False
        )
        assert generator is not None

    def test_kbc_chunk_generator_batch_forward(self):
        """Test KBCChunkGeneratorBatch forward processing"""
        # Check if dependencies are available
               
        # Create test text file
        test_text = "Sample text for batch chunking. " * 30
        test_file = os.path.join(self.temp_dir, "batch_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
        
        generator = kbc.KBCChunkGeneratorBatch(
            chunk_size=128,
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [{'text_path': test_file}]
        results = generator(input_data)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'chunk_path' in results[0]
        # Check if chunk file was created
        if results[0]['chunk_path']:
            assert os.path.exists(results[0]['chunk_path'])

    # ============ KBCTextCleaner Tests ============
    
    def test_kbc_text_cleaner_init(self):
        """Test KBCTextCleaner initialization"""
        cleaner = kbc.KBCTextCleaner(
            llm=None,
            lang="zh",
            _save_data=False
        )
        assert cleaner is not None
        assert cleaner.lang == "zh"

    def test_kbc_text_cleaner_forward_with_mock(self):
        """Test KBCTextCleaner forward with mocked LLM"""
        # Create mock LLM
        mock_llm = Mock()
        mock_serve = Mock()
        mock_serve.return_value = "<cleaned_start>Cleaned test content<cleaned_end>"
        mock_llm.share.return_value = mock_serve
        
        cleaner = kbc.KBCTextCleaner(
            llm=mock_llm,
            lang="en",
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [{'raw_chunk': 'Test content with <html>tags</html>'}]
        results = cleaner(input_data)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'cleaned_chunk' in results[0]
        assert results[0]['cleaned_chunk'] == 'Cleaned test content'

    def test_kbc_text_cleaner_multiple_items(self):
        """Test KBCTextCleaner with multiple items"""
        mock_llm = Mock()
        mock_serve = Mock()
        mock_serve.side_effect = [
            "<cleaned_start>Clean 1<cleaned_end>",
            "<cleaned_start>Clean 2<cleaned_end>",
            "<cleaned_start>Clean 3<cleaned_end>"
        ]
        mock_llm.share.return_value = mock_serve
        
        cleaner = kbc.KBCTextCleaner(
            llm=mock_llm,
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [
            {'raw_chunk': 'Content 1'},
            {'raw_chunk': 'Content 2'},
            {'raw_chunk': 'Content 3'}
        ]
        results = cleaner(input_data)
        
        assert len(results) == 3
        assert all('cleaned_chunk' in item for item in results)

    # ============ KBCTextCleanerBatch Tests ============
    
    def test_kbc_text_cleaner_batch_init(self):
        """Test KBCTextCleanerBatch initialization"""
        cleaner = kbc.KBCTextCleanerBatch(
            llm=None,
            lang="zh",
            _save_data=False
        )
        assert cleaner is not None

    def test_kbc_text_cleaner_batch_forward(self):
        """Test KBCTextCleanerBatch forward with file processing"""
        # Create test chunk file
        chunk_data = [
            {"raw_chunk": "Test chunk 1 with <html>tags</html>"},
            {"raw_chunk": "Test chunk 2 with special chars @#$"}
        ]
        chunk_file = os.path.join(self.temp_dir, "test_chunks.json")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False)
        
        # Mock LLM
        mock_llm = Mock()
        mock_serve = Mock()
        mock_serve.side_effect = [
            "<cleaned_start>Clean 1<cleaned_end>",
            "<cleaned_start>Clean 2<cleaned_end>"
        ]
        mock_llm.share.return_value = mock_serve
        
        cleaner = kbc.KBCTextCleanerBatch(
            llm=mock_llm,
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [{'chunk_path': chunk_file}]
        results = cleaner(input_data)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'cleaned_chunk_path' in results[0]
        
        # Verify file was updated
        with open(chunk_file, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
        assert all('cleaned_chunk' in item for item in updated_data)

    # # ============ FileOrURLToMarkdownConverterBatch Tests ============

    # def test_file_converter_batch_init(self):
    #     """Test FileOrURLToMarkdownConverterBatch initialization"""
    #     converter = kbc.FileOrURLToMarkdownConverterBatch(
    #         intermediate_dir=os.path.join(self.temp_dir, "converter_cache"),
    #         _save_data=False
    #     )
    #     assert converter is not None

    # def test_file_converter_batch_txt_file(self):
    #     """Test converting TXT file"""
    #     # Create test TXT file
    #     test_file = os.path.join(self.temp_dir, "test.txt")
    #     with open(test_file, 'w', encoding='utf-8') as f:
    #         f.write("Test plain text content")
        
    #     converter = kbc.FileOrURLToMarkdownConverterBatch(
    #         intermediate_dir=os.path.join(self.temp_dir, "cache"),
    #         _save_data=False,
    #         _concurrency_mode='single'
    #     )
        
    #     input_data = [{'source': test_file}]
    #     results = converter(input_data)
        
    #     assert isinstance(results, list)
    #     assert len(results) == 1
    #     assert 'text_path' in results[0]
    #     # TXT files should be passed through
    #     assert results[0]['text_path'] == test_file

    # def test_file_converter_batch_md_file(self):
    #     """Test converting MD file"""
    #     test_file = os.path.join(self.temp_dir, "test.md")
    #     with open(test_file, 'w', encoding='utf-8') as f:
    #         f.write("# Test Markdown\n\nContent here")
        
    #     converter = kbc.FileOrURLToMarkdownConverterBatch(
    #         intermediate_dir=os.path.join(self.temp_dir, "cache"),
    #         _save_data=False,
    #         _concurrency_mode='single'
    #     )
        
    #     input_data = [{'source': test_file}]
    #     results = converter(input_data)
        
    #     assert len(results) == 1
    #     assert results[0]['text_path'] == test_file

    # def test_file_converter_batch_invalid_file(self):
    #     """Test with non-existent file"""
    #     converter = kbc.FileOrURLToMarkdownConverterBatch(
    #         intermediate_dir=os.path.join(self.temp_dir, "cache"),
    #         _save_data=False,
    #         _concurrency_mode='single',
    #         _ignore_errors=True
    #     )
        
    #     input_data = [{'source': '/nonexistent/file.pdf'}]
    #     results = converter(input_data)
        
    #     assert isinstance(results, list)

    # ============ FileOrURLToMarkdownConverterAPI Tests ============

    def test_file_converter_api_init(self):
        """Test FileOrURLToMarkdownConverterAPI initialization"""
        converter = kbc.FileOrURLToMarkdownConverterAPI(
            intermediate_dir=os.path.join(self.temp_dir, "api_cache"),
            mineru_url="http://localhost:20234",
            mineru_backend="vlm-vllm-async-engine",
            upload_mode=True,
            _save_data=False
        )
        assert converter is not None
        assert converter.mineru_url == "http://localhost:20234"
        assert converter.mineru_backend == "vlm-vllm-async-engine"
        assert converter.upload_mode is True

    def test_file_converter_api_txt_file(self):
        """Test FileOrURLToMarkdownConverterAPI with TXT file (pass-through)"""
        test_file = os.path.join(self.temp_dir, "api_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test plain text content for API converter")
        
        converter = kbc.FileOrURLToMarkdownConverterAPI(
            intermediate_dir=os.path.join(self.temp_dir, "api_cache"),
            mineru_url="http://localhost:20234",
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [{'source': test_file}]
        results = converter(input_data)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'text_path' in results[0]
        # TXT files should be passed through
        assert results[0]['text_path'] == test_file

    def test_file_converter_api_md_file(self):
        """Test FileOrURLToMarkdownConverterAPI with MD file (pass-through)"""
        test_file = os.path.join(self.temp_dir, "api_test.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("# API Test Markdown\n\nContent here")
        
        converter = kbc.FileOrURLToMarkdownConverterAPI(
            intermediate_dir=os.path.join(self.temp_dir, "api_cache"),
            mineru_url="http://localhost:20234",
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [{'source': test_file}]
        results = converter(input_data)
        
        assert len(results) == 1
        assert results[0]['text_path'] == test_file

    def test_file_converter_api_missing_url(self):
        """Test FileOrURLToMarkdownConverterAPI raises error when mineru_url is missing"""
        converter = kbc.FileOrURLToMarkdownConverterAPI(
            intermediate_dir=os.path.join(self.temp_dir, "api_cache"),
            mineru_url=None,
            _save_data=False
        )
        
        test_file = os.path.join(self.temp_dir, "test.pdf")
        with open(test_file, 'wb') as f:
            f.write(b'%PDF-1.4 fake pdf content')
        
        input_data = [{'source': test_file}]
        
        # Should raise ValueError because mineru_url is not set
        with pytest.raises(ValueError, match="mineru_url is required"):
            converter(input_data)

    # ============ KBCMultiHopQAGeneratorBatch Tests ============

    def test_multihop_qa_generator_init(self):
        """Test KBCMultiHopQAGeneratorBatch initialization"""
        generator = kbc.KBCMultiHopQAGeneratorBatch(
            llm=None,
            lang="zh",
            _save_data=False
        )
        assert generator is not None

    def test_multihop_qa_generator_forward(self):
        """Test MultiHopQAGenerator forward processing"""
        # Create test chunk file with cleaned content
        chunk_data = [
            {"cleaned_chunk": "Python is a high-level programming language. It was created by Guido van Rossum. Python emphasizes code readability."},
            {"cleaned_chunk": "Machine learning is a subset of AI. It enables systems to learn from data. Deep learning uses neural networks."}
        ]
        chunk_file = os.path.join(self.temp_dir, "qa_test_chunks.json")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False)
        
        # Mock LLM responses
        mock_llm = Mock()
        mock_serve = Mock()
        mock_serve.side_effect = [
            '{"question": "Who created Python?", "answer": "Guido van Rossum"}',
            '{"question": "What is machine learning?", "answer": "A subset of AI"}'
        ]
        mock_llm.share.return_value = mock_serve
        
        generator = kbc.KBCMultiHopQAGeneratorBatch(
            llm=mock_llm,
            lang="en",
            _save_data=False,
            _concurrency_mode='single'
        )
        
        input_data = [{'chunk_path': chunk_file}]
        results = generator(input_data)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'enhanced_chunk_path' in results[0]

    # ============ QAExtractor Tests ============
    
    def test_qa_extractor_init(self):
        """Test QAExtractor initialization"""
        extractor = kbc.QAExtractor(
            input_qa_key="QA_pairs",
            input_instruction="Answer the question.",
            _save_data=False
        )
        assert extractor is not None

    def test_qa_extractor_with_qa_pairs(self):
        """Test QAExtractor with QA_pairs data"""
        extractor = kbc.QAExtractor(_save_data=False)
        data = [
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "What is Python?", "answer": "A programming language."},
                        {"question": "What is LazyLLM?", "answer": "A framework for LLM."},
                    ]
                }
            },
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "What is AI?", "answer": "Artificial Intelligence."},
                    ]
                }
            }
        ]
        results = extractor(data)
        assert len(results) == 3
        assert all("instruction" in r for r in results)
        assert all("input" in r for r in results)
        assert all("output" in r for r in results)

    def test_qa_extractor_empty_qa_pairs(self):
        """Test QAExtractor with empty QA_pairs"""
        extractor = kbc.QAExtractor(_save_data=False)
        data = [
            {"QA_pairs": {"qa_pairs": []}},
        ]
        results = extractor(data)
        assert len(results) == 0

    def test_qa_extractor_invalid_qa_format(self):
        """Test QAExtractor with invalid QA format"""
        extractor = kbc.QAExtractor(_save_data=False)
        data = [
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "", "answer": "Test A."},  # Empty question
                        {"question": "Test Q?", "answer": ""},  # Empty answer
                        {"question": "Valid Q?", "answer": "Valid A."},  # Valid
                    ]
                }
            }
        ]
        results = extractor(data)
        # Only valid QA pairs should be extracted
        assert len(results) == 1
        assert results[0]["input"] == "Valid Q?"

    def test_qa_extractor_with_output_file(self):
        """Test QAExtractor with output JSON file"""
        output_file = os.path.join(self.temp_dir, "qa_output.json")
        extractor = kbc.QAExtractor(output_json_file=output_file, _save_data=False)

        data = [
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "Q1?", "answer": "A1."},
                        {"question": "Q2?", "answer": "A2."},
                    ]
                }
            }
        ]
        results = extractor(data)

        # Check file was created
        assert os.path.exists(output_file)

        # Check file contents
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        assert len(saved_data) == 2
        assert saved_data[0]["input"] == "Q1?"
        assert saved_data[1]["input"] == "Q2?"

    def test_qa_extractor_custom_output_keys(self):
        """Test QAExtractor with custom output keys"""
        extractor = kbc.QAExtractor(_save_data=False)
        
        data = [
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "Test Q?", "answer": "Test A."}
                    ]
                }
            }
        ]
        
        # Use forward_batch_input directly to pass custom parameters
        # because __call__() only accepts inputs, not additional kwargs
        results = extractor.forward_batch_input(
            data,
            output_instruction_key="custom_instruction",
            output_question_key="custom_question",
            output_answer_key="custom_answer"
        )
        
        assert len(results) == 1
        assert "custom_instruction" in results[0]
        assert "custom_question" in results[0]
        assert "custom_answer" in results[0]

    # ============ Integration Tests ============
    
    def test_pipeline_chunk_and_clean(self):
        """Test pipeline: chunk then clean"""
        # Create test file
        test_text = "This is test content. " * 20
        test_file = os.path.join(self.temp_dir, "pipeline_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
        
        # Step 1: Chunk
        chunker = kbc.KBCChunkGenerator(
            chunk_size=64,
            _save_data=False,
            _concurrency_mode='single'
        )
        chunks = chunker([{'text_path': test_file}])
        
        assert len(chunks) > 1
        assert all('raw_chunk' in item for item in chunks)
        
        # Step 2: Clean (with mock)
        mock_llm = Mock()
        mock_serve = Mock()
        mock_serve.side_effect = [f"<cleaned_start>Clean {i}<cleaned_end>" for i in range(len(chunks))]
        mock_llm.share.return_value = mock_serve
        
        cleaner = kbc.KBCTextCleaner(
            llm=mock_llm,
            _save_data=False,
            _concurrency_mode='single'
        )
        cleaned = cleaner(chunks)
        
        assert len(cleaned) == len(chunks)
        assert all('cleaned_chunk' in item for item in cleaned)

    def test_error_handling_with_ignore(self):
        """Test error handling with _ignore_errors=True"""
        generator = kbc.KBCChunkGenerator(
            _save_data=False,
            _concurrency_mode='single',
            _ignore_errors=True
        )
        
        # Mix valid and invalid paths
        input_data = [
            {'text_path': '/invalid/path1.txt'},
            {'text_path': '/invalid/path2.txt'}
        ]
        
        # Should not raise exception
        results = generator(input_data)
        assert isinstance(results, list)

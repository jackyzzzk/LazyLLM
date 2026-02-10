"""Test cases for Reranker Synthesis operators"""
import pytest
import json
import tempfile
import os
import shutil
from unittest.mock import Mock
import lazyllm
from lazyllm import config
from lazyllm.tools.data import reranker



class TestRerankerSynthesisOperators:
    """Test suite for Reranker Synthesis operators"""

    def setup_method(self):
        """Setup test environment"""
        self.root_dir = './test_data_op_reranker'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def model_set(self):
        """Create mock LLM for testing"""
        mock_llm = Mock()
        mock_llm.share.return_value = mock_llm
        mock_llm.start.return_value = None
        # Use online chat model for testing
        self.llm = lazyllm.OnlineChatModule()
        return mock_llm

    # ============ RerankerQueryGenerator Tests ============

    def test_query_generator_init(self):
        """Test RerankerQueryGenerator initialization"""
        generator = reranker.RerankerQueryGenerator(
            llm_serving=None,
            num_queries=5,
            lang="zh",
            _save_data=False
        )
        assert generator is not None

    def test_query_generator_init_english(self):
        """Test RerankerQueryGenerator initialization with English"""
        generator = reranker.RerankerQueryGenerator(
            llm_serving=None,
            num_queries=3,
            lang="en",
            _save_data=False
        )
        assert generator is not None

    def test_query_generator_init_with_difficulty_levels(self):
        """Test RerankerQueryGenerator initialization with custom difficulty levels"""
        generator = reranker.RerankerQueryGenerator(
            llm_serving=None,
            difficulty_levels=["easy", "hard"],
            _save_data=False
        )
        assert generator is not None
        assert generator.difficulty_levels == ["easy", "hard"]

    def test_query_generator_forward_with_mock_llm(self):
        """Test RerankerQueryGenerator forward with mock LLM"""
        mock_llm = self.model_set()
        # Mock LLM returns JSON with queries
        mock_llm.return_value = json.dumps({
            "queries": [
                {"query": "什么是人工智能？", "difficulty": "easy"},
                {"query": "AI如何影响未来？", "difficulty": "medium"}
            ]
        })

        generator = reranker.RerankerQueryGenerator(
            llm_serving=mock_llm,
            num_queries=2,
            lang="zh",
            _save_data=False,
            _concurrency_mode='single'
        )

        input_data = [{'passage': '人工智能是计算机科学的一个分支...'}]
        results = generator(input_data)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all('query' in item for item in results)
        assert all('pos' in item for item in results)
        assert all('difficulty' in item for item in results)

    # ============ RerankerHardNegativeMiner Tests ============

    def test_hard_negative_miner_init_random(self):
        """Test RerankerHardNegativeMiner initialization with random strategy"""
        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="random",
            num_negatives=7,
            _save_data=False
        )
        assert miner is not None
        assert miner.mining_strategy == "random"

    def test_hard_negative_miner_init_bm25(self):
        """Test RerankerHardNegativeMiner initialization with BM25 strategy"""
        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="bm25",
            num_negatives=7,
            _save_data=False
        )
        assert miner is not None

    def test_hard_negative_miner_init_semantic(self):
        """Test RerankerHardNegativeMiner initialization with semantic strategy"""
        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="semantic",
            num_negatives=7,
            embedding_serving=None,
            _save_data=False
        )
        assert miner is not None

    def test_hard_negative_miner_init_mixed(self):
        """Test RerankerHardNegativeMiner initialization with mixed strategy"""
        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="mixed",
            num_negatives=7,
            bm25_ratio=0.5,
            _save_data=False
        )
        assert miner is not None

    def test_hard_negative_miner_random_strategy(self):
        """Test RerankerHardNegativeMiner with random strategy"""
        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="random",
            num_negatives=3,
            seed=42,
            _save_data=False
        )

        data = [
            {"query": "什么是AI？", "pos": ["人工智能是..."]},
            {"query": "什么是ML？", "pos": ["机器学习是..."]},
            {"query": "什么是DL？", "pos": ["深度学习是..."]},
        ]
        # Add more passages to corpus
        corpus = ["人工智能是...", "机器学习是...", "深度学习是...",
                  "自然语言处理是...", "计算机视觉是...", "强化学习是..."]

        results = miner(data, corpus=corpus)

        assert len(results) == 3
        assert all('neg' in item for item in results)
        # Each should have up to 3 negatives
        for item in results:
            assert len(item['neg']) <= 3

    def test_hard_negative_miner_bm25_strategy(self):
        """Test RerankerHardNegativeMiner with BM25 strategy"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")

        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="bm25",
            num_negatives=2,
            _save_data=False
        )

        data = [
            {"query": "什么是人工智能", "pos": ["人工智能是计算机科学的分支"]},
        ]
        corpus = [
            "人工智能是计算机科学的分支",
            "人工智能技术正在快速发展",
            "机器学习是人工智能的子领域",
            "深度学习需要大量数据",
            "自然语言处理是AI应用"
        ]

        results = miner(data, corpus=corpus)

        assert len(results) == 1
        assert 'neg' in results[0]
        # BM25 should return lexically similar but not positive samples
        assert results[0]['pos'][0] not in results[0]['neg']

    # ============ RerankerDataFormatter Tests ============

    def test_data_formatter_init_flagreranker(self):
        """Test RerankerDataFormatter initialization with flagreranker format"""
        formatter = reranker.RerankerDataFormatter(
            output_format="flagreranker",
            train_group_size=8,
            _save_data=False
        )
        assert formatter is not None

    def test_data_formatter_init_cross_encoder(self):
        """Test RerankerDataFormatter initialization with cross_encoder format"""
        formatter = reranker.RerankerDataFormatter(
            output_format="cross_encoder",
            _save_data=False
        )
        assert formatter is not None

    def test_data_formatter_init_pairwise(self):
        """Test RerankerDataFormatter initialization with pairwise format"""
        formatter = reranker.RerankerDataFormatter(
            output_format="pairwise",
            _save_data=False
        )
        assert formatter is not None

    def test_data_formatter_flagreranker_format(self):
        """Test formatting to flagreranker format"""
        formatter = reranker.RerankerDataFormatter(
            output_format="flagreranker",
            train_group_size=4,  # 1 pos + 3 neg
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "什么是AI？", "pos": ["人工智能是..."], "neg": ["天气是...", "美食是...", "音乐是..."]},
        ]
        results = formatter(data)

        assert len(results) == 1
        assert results[0]["query"] == "什么是AI？"
        assert results[0]["pos"] == ["人工智能是..."]
        assert len(results[0]["neg"]) == 3  # train_group_size - 1

    def test_data_formatter_cross_encoder_format(self):
        """Test formatting to cross_encoder format"""
        formatter = reranker.RerankerDataFormatter(
            output_format="cross_encoder",
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        results = formatter(data)

        # Should expand: 1 pos with label 1 + 2 neg with label 0 = 3 rows
        assert len(results) == 3
        assert all("label" in r for r in results)
        assert sum(r["label"] for r in results) == 1  # Only 1 positive

    def test_data_formatter_pairwise_format(self):
        """Test formatting to pairwise format"""
        formatter = reranker.RerankerDataFormatter(
            output_format="pairwise",
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        results = formatter(data)

        # Should expand: 1 pos * 2 neg = 2 pairwise
        assert len(results) == 2
        assert all("doc_pos" in r and "doc_neg" in r for r in results)

    def test_data_formatter_with_output_file(self):
        """Test formatting with output file"""
        output_file = os.path.join(self.temp_dir, "output.jsonl")
        formatter = reranker.RerankerDataFormatter(
            output_format="flagreranker",
            output_file=output_file,
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]},
        ]
        results = formatter(data)

        # Check file was created
        assert os.path.exists(output_file)

    def test_data_formatter_neg_padding(self):
        """Test that negatives are padded when insufficient"""
        formatter = reranker.RerankerDataFormatter(
            output_format="flagreranker",
            train_group_size=8,  # Need 7 negatives
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},  # Only 2 negatives
        ]
        results = formatter(data)

        assert len(results) == 1
        assert len(results[0]["neg"]) == 7  # Padded to 7

    # ============ RerankerTrainTestSplitter Tests ============

    def test_train_test_splitter_init(self):
        """Test RerankerTrainTestSplitter initialization"""
        splitter = reranker.RerankerTrainTestSplitter(
            test_size=0.1,
            seed=42,
            _save_data=False
        )
        assert splitter is not None

    def test_train_test_splitter_split(self):
        """Test train/test splitting"""
        splitter = reranker.RerankerTrainTestSplitter(
            test_size=0.3,
            seed=42,
            _save_data=False
        )
        data = [
            {"query": f"query_{i}", "pos": [f"pos_{i}"], "neg": [f"neg_{i}"]}
            for i in range(10)
        ]
        results = splitter(data)

        # All samples should have split label
        assert len(results) == 10
        assert all("split" in r for r in results)

        # Check split ratio
        train_count = sum(1 for r in results if r["split"] == "train")
        test_count = sum(1 for r in results if r["split"] == "test")
        assert train_count == 7  # 70%
        assert test_count == 3   # 30%

    def test_train_test_splitter_with_output_files(self):
        """Test splitting with output files"""
        train_file = os.path.join(self.temp_dir, "train.jsonl")
        test_file = os.path.join(self.temp_dir, "eval.jsonl")

        splitter = reranker.RerankerTrainTestSplitter(
            test_size=0.2,
            seed=42,
            train_output_file=train_file,
            test_output_file=test_file,
            _save_data=False
        )
        data = [
            {"query": f"query_{i}", "pos": [f"pos_{i}"], "neg": [f"neg_{i}"]}
            for i in range(10)
        ]
        results = splitter(data)

        # Check files were created
        assert os.path.exists(train_file)
        assert os.path.exists(test_file)

        # Verify file contents
        with open(train_file, 'r') as f:
            train_lines = f.readlines()
        with open(test_file, 'r') as f:
            test_lines = f.readlines()

        assert len(train_lines) == 8  # 80%
        assert len(test_lines) == 2   # 20%

    # ============ RerankerFromEmbeddingConverter Tests ============

    def test_from_embedding_converter_init(self):
        """Test RerankerFromEmbeddingConverter initialization"""
        converter = reranker.RerankerFromEmbeddingConverter(
            adjust_neg_count=7,
            _save_data=False
        )
        assert converter is not None

    def test_from_embedding_converter_convert_data(self):
        """Test converting embedding data to reranker format"""
        converter = reranker.RerankerFromEmbeddingConverter(
            adjust_neg_count=3,
            _save_data=False,
            _concurrency_mode='single'
        )
        # Embedding format with prompt
        data = [
            {
                "query": "Q1",
                "pos": ["P1"],
                "neg": ["N1", "N2", "N3", "N4", "N5"],
                "prompt": "Represent this sentence: "
            },
        ]
        results = converter(data)

        assert len(results) == 1
        assert results[0]["query"] == "Q1"
        assert results[0]["pos"] == ["P1"]
        assert len(results[0]["neg"]) == 3  # Adjusted to 3
        assert "prompt" not in results[0]  # No prompt in reranker format

    def test_from_embedding_converter_neg_padding(self):
        """Test that negatives are padded when insufficient"""
        converter = reranker.RerankerFromEmbeddingConverter(
            adjust_neg_count=5,
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},  # Only 2 negatives
        ]
        results = converter(data)

        assert len(results) == 1
        assert len(results[0]["neg"]) == 5  # Padded to 5

    def test_from_embedding_converter_with_output_file(self):
        """Test converting with output file"""
        output_file = os.path.join(self.temp_dir, "reranker.jsonl")
        converter = reranker.RerankerFromEmbeddingConverter(
            adjust_neg_count=7,
            output_file=output_file,
            _save_data=False,
            _concurrency_mode='single'
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        results = converter(data)

        # Check file was created
        assert os.path.exists(output_file)

    def test_from_embedding_converter_empty_query(self):
        """Test handling of empty query"""
        converter = reranker.RerankerFromEmbeddingConverter(
            adjust_neg_count=3,
            _save_data=False,
            _concurrency_mode='single',
            _ignore_errors=True
        )
        data = [
            {"query": "", "pos": ["P1"], "neg": ["N1"]},  # Empty query
            {"query": "Q2", "pos": ["P2"], "neg": ["N2"]},
        ]
        results = converter(data)

        # Empty query should be filtered out (returns [])
        # Valid query should be converted
        valid_results = [r for r in results if r]
        assert len(valid_results) == 1
        assert valid_results[0]["query"] == "Q2"

    # ============ Pipeline Tests ============

    def test_pipeline_query_generation_to_formatting(self):
        """Test pipeline from query generation to formatting"""
        mock_llm = self.model_set()
        mock_llm.return_value = json.dumps({
            "queries": [
                {"query": "什么是AI？", "difficulty": "easy"},
            ]
        })

        # Step 1: Generate queries
        generator = reranker.RerankerQueryGenerator(
            llm_serving=mock_llm,
            num_queries=1,
            _save_data=False,
            _concurrency_mode='single'
        )
        passages = [{'passage': '人工智能是...'}]
        query_results = generator(passages)

        assert len(query_results) == 1
        assert 'query' in query_results[0]
        assert 'pos' in query_results[0]

        # Step 2: Mine negatives (random strategy for simplicity)
        miner = reranker.RerankerHardNegativeMiner(
            mining_strategy="random",
            num_negatives=2,
            _save_data=False
        )
        corpus = ["人工智能是...", "机器学习是...", "深度学习是..."]
        mined_results = miner(query_results, corpus=corpus)

        assert len(mined_results) == 1
        assert 'neg' in mined_results[0]

        # Step 3: Format data
        formatter = reranker.RerankerDataFormatter(
            output_format="flagreranker",
            train_group_size=3,
            _save_data=False,
            _concurrency_mode='single'
        )
        formatted_results = formatter(mined_results)

        assert len(formatted_results) == 1
        assert 'query' in formatted_results[0]
        assert 'pos' in formatted_results[0]
        assert 'neg' in formatted_results[0]

    def test_pipeline_embedding_to_reranker_full(self):
        """Test full pipeline from embedding data to reranker training data"""
        # Step 1: Convert embedding data
        converter = reranker.RerankerFromEmbeddingConverter(
            adjust_neg_count=3,
            _save_data=False,
            _concurrency_mode='single'
        )
        embedding_data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2", "N3"], "prompt": "..."},
            {"query": "Q2", "pos": ["P2"], "neg": ["N4", "N5", "N6"], "prompt": "..."},
        ]
        converted = converter(embedding_data)

        assert len(converted) == 2
        assert all("prompt" not in item for item in converted)

        # Step 2: Format to cross_encoder
        formatter = reranker.RerankerDataFormatter(
            output_format="cross_encoder",
            _save_data=False,
            _concurrency_mode='single'
        )
        formatted = formatter(converted)

        # Each sample should expand to 1 pos + 3 neg = 4 items
        assert len(formatted) == 8  # 2 samples * 4 items each

        # Step 3: Split train/test
        splitter = reranker.RerankerTrainTestSplitter(
            test_size=0.25,
            seed=42,
            _save_data=False
        )
        split_results = splitter(formatted)

        assert len(split_results) == 8
        train_count = sum(1 for r in split_results if r["split"] == "train")
        test_count = sum(1 for r in split_results if r["split"] == "test")
        assert train_count + test_count == 8

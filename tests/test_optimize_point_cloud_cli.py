"""
Tests for optimize_point_cloud.py CLI functionality using mocking.

This demonstrates how to test CLI scripts that use the new StorageConfig
parameter parsing without actually running the optimization pipeline.
"""

import unittest
import argparse
import tempfile
import os
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path

# Add the project root to the path so we can import optimize_point_cloud
sys.path.insert(0, str(Path(__file__).parent.parent))
import optimize_point_cloud
from pocaduck.storage_config import StorageConfig


class TestOptimizePointCloudCLI(unittest.TestCase):
    """Test CLI functionality of optimize_point_cloud.py script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_base_path = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.test_base_path, exist_ok=True)
        
        # Create a mock unified index file
        self.mock_index_path = os.path.join(self.test_base_path, "unified_index.db")
        with open(self.mock_index_path, 'w') as f:
            f.write("mock database file")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('optimize_point_cloud.shard_labels')
    @patch('optimize_point_cloud.StorageConfig.from_args')
    def test_shard_action_cli_parsing(self, mock_from_args, mock_shard_labels):
        """Test that shard action correctly parses CLI arguments."""
        # Setup mocks
        mock_config = MagicMock()
        mock_from_args.return_value = mock_config
        mock_shard_labels.return_value = ['shard1.txt', 'shard2.txt']
        
        # Simulate command line arguments
        test_args = [
            '--action', 'shard',
            '--base-path', self.test_base_path,
            '--num-shards', '4',
            '--shard-output-dir', self.temp_dir
        ]
        
        # Mock sys.argv and run main
        with patch('sys.argv', ['optimize_point_cloud.py'] + test_args):
            optimize_point_cloud.main()
        
        # Verify StorageConfig.from_args was called with parsed arguments
        mock_from_args.assert_called_once()
        args = mock_from_args.call_args[0][0]
        self.assertEqual(args.base_path, self.test_base_path)
        self.assertEqual(args.action, 'shard')
        self.assertEqual(args.num_shards, 4)
        
        # Verify shard_labels was called with correct parameters
        mock_shard_labels.assert_called_once_with(
            storage_config=mock_config,
            source_index_path=self.mock_index_path,
            num_shards=4,
            output_dir=self.temp_dir,
            verbose=True
        )
    
    @patch('optimize_point_cloud.optimize_point_clouds')
    @patch('optimize_point_cloud.StorageConfig.from_args')
    def test_optimize_action_with_s3_storage(self, mock_from_args, mock_optimize):
        """Test optimize action with S3 storage parameters."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.storage_type = 's3'
        mock_from_args.return_value = mock_config
        
        # Create a mock labels file
        labels_file = os.path.join(self.temp_dir, "labels_shard_0.txt")
        with open(labels_file, 'w') as f:
            f.write("12345\n67890\n")
        
        # Simulate command line arguments with S3 storage
        test_args = [
            '--action', 'optimize',
            '--base-path', 's3://test-bucket/data',
            '--s3-region', 'us-west-2',
            '--s3-access-key-id', 'test-key',
            '--labels-file', labels_file,
            '--worker-id', 'test-worker',
            '--threads', '8'
        ]
        
        with patch('sys.argv', ['optimize_point_cloud.py'] + test_args):
            optimize_point_cloud.main()
        
        # Verify StorageConfig.from_args was called
        mock_from_args.assert_called_once()
        args = mock_from_args.call_args[0][0]
        self.assertEqual(args.base_path, 's3://test-bucket/data')
        self.assertEqual(args.s3_region, 'us-west-2')
        self.assertEqual(args.s3_access_key_id, 'test-key')
        
        # Verify optimize_point_clouds was called with correct parameters
        mock_optimize.assert_called_once()
        call_kwargs = mock_optimize.call_args[1]
        self.assertEqual(call_kwargs['storage_config'], mock_config)
        self.assertEqual(call_kwargs['worker_id'], 'test-worker')
        self.assertEqual(call_kwargs['threads'], 8)
        self.assertEqual(call_kwargs['labels_to_process'], [12345, 67890])
    
    @patch('optimize_point_cloud.consolidate_optimized_indices')
    @patch('optimize_point_cloud.StorageConfig.from_args')
    def test_consolidate_action_cli_parsing(self, mock_from_args, mock_consolidate):
        """Test consolidate action CLI parsing."""
        mock_config = MagicMock()
        mock_from_args.return_value = mock_config
        
        test_args = [
            '--action', 'consolidate',
            '--base-path', self.test_base_path,
            '--target-path', os.path.join(self.temp_dir, 'optimized'),
            '--threads', '16',
            '--quiet'
        ]
        
        with patch('sys.argv', ['optimize_point_cloud.py'] + test_args):
            optimize_point_cloud.main()
        
        # Verify consolidate was called with correct parameters
        mock_consolidate.assert_called_once_with(
            storage_config=mock_config,
            target_path=os.path.join(self.temp_dir, 'optimized'),
            target_index_path=None,
            threads=16,
            verbose=False  # --quiet flag should set verbose=False
        )
    
    @patch('optimize_point_cloud.StorageConfig.from_args')
    def test_storage_config_validation_error(self, mock_from_args):
        """Test that StorageConfig validation errors are properly handled."""
        # Make StorageConfig.from_args raise a validation error
        mock_from_args.side_effect = ValueError("Conflicting storage parameters provided")
        
        test_args = [
            '--action', 'shard',
            '--base-path', '/tmp/test',
            '--s3-region', 'us-west-2',  # This conflicts with local path
            '--num-shards', '2'
        ]
        
        with patch('sys.argv', ['optimize_point_cloud.py'] + test_args):
            with self.assertRaises(ValueError) as cm:
                optimize_point_cloud.main()
            
            self.assertIn("Conflicting storage parameters", str(cm.exception))
    
    def test_argument_parser_setup(self):
        """Test that the argument parser is set up correctly."""
        parser = argparse.ArgumentParser(description="Test")
        StorageConfig.add_storage_args(parser)
        
        # Test that all expected arguments are added
        help_text = parser.format_help()
        
        # Check for storage arguments
        self.assertIn("--base-path", help_text)
        self.assertIn("S3 Storage Options", help_text)
        self.assertIn("--s3-region", help_text)
        self.assertIn("--s3-access-key-id", help_text)
        self.assertIn("Google Cloud Storage Options", help_text)
        self.assertIn("--gcs-project-id", help_text)
        self.assertIn("Azure Storage Options", help_text)
        self.assertIn("--azure-connection-string", help_text)
    
    @patch('optimize_point_cloud.optimize_point_clouds')
    @patch('optimize_point_cloud.StorageConfig.from_args')
    def test_labels_parsing_from_command_line(self, mock_from_args, mock_optimize):
        """Test parsing labels from command line argument."""
        mock_config = MagicMock()
        mock_from_args.return_value = mock_config
        
        test_args = [
            '--action', 'optimize',
            '--base-path', self.test_base_path,
            '--labels', '123,456,789',
            '--worker-id', 'test'
        ]
        
        with patch('sys.argv', ['optimize_point_cloud.py'] + test_args):
            optimize_point_cloud.main()
        
        # Verify labels were parsed correctly
        call_kwargs = mock_optimize.call_args[1]
        self.assertEqual(call_kwargs['labels_to_process'], [123, 456, 789])
    
    def test_missing_required_arguments(self):
        """Test error handling for missing required arguments."""
        # Missing --base-path should cause parser error
        with patch('sys.argv', ['optimize_point_cloud.py', '--action', 'shard']):
            with patch('sys.stderr'):  # Suppress error output
                with self.assertRaises(SystemExit):
                    optimize_point_cloud.main()
        
        # Missing --num-shards for shard action should cause error
        with patch('sys.argv', ['optimize_point_cloud.py', '--action', 'shard', '--base-path', '/tmp']):
            with patch('sys.stderr'):  # Suppress error output
                with self.assertRaises(SystemExit):
                    optimize_point_cloud.main()


class TestStorageConfigMockingStrategies(unittest.TestCase):
    """Demonstrate different mocking strategies for testing StorageConfig."""
    
    def test_mock_storage_config_creation(self):
        """Show how to mock StorageConfig for isolated testing."""
        with patch('pocaduck.storage_config.StorageConfig') as MockStorageConfig:
            # Create a mock instance
            mock_instance = MagicMock()
            mock_instance.base_path = '/test/path'
            mock_instance.storage_type = 'local'
            MockStorageConfig.return_value = mock_instance
            
            # Test code that uses StorageConfig
            config = StorageConfig(base_path='/test/path')
            self.assertEqual(config.base_path, '/test/path')
            self.assertEqual(config.storage_type, 'local')
    
    def test_mock_duckdb_config(self):
        """Show how to mock DuckDB configuration for testing."""
        config = StorageConfig(base_path='/test/path')
        
        with patch.object(config, 'get_duckdb_config') as mock_get_config:
            mock_get_config.return_value = {'test_setting': 'test_value'}
            
            duckdb_config = config.get_duckdb_config()
            self.assertEqual(duckdb_config, {'test_setting': 'test_value'})
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    def test_mock_environment_variables(self):
        """Show how to mock environment variables for testing."""
        # This demonstrates testing with environment variables
        # even though our current StorageConfig doesn't use them directly
        self.assertEqual(os.environ['AWS_REGION'], 'us-east-1')


if __name__ == '__main__':
    # Run with verbose output to see test descriptions
    unittest.main(verbosity=2)
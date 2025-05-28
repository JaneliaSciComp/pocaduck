"""
Tests for StorageConfig CLI parameter parsing functionality.

This module tests the new add_storage_args() and from_args() class methods
to ensure proper CLI parameter handling and validation.
"""

import unittest
import argparse
from unittest.mock import patch, MagicMock
import tempfile
import os

from pocaduck.storage_config import StorageConfig


class TestStorageConfigCLI(unittest.TestCase):
    """Test CLI parameter parsing for StorageConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = argparse.ArgumentParser()
        StorageConfig.add_storage_args(self.parser)
    
    def test_local_storage_basic(self):
        """Test basic local storage configuration."""
        args = self.parser.parse_args(['--base-path', '/tmp/test'])
        config = StorageConfig.from_args(args)
        
        self.assertEqual(config.base_path, '/tmp/test')
        self.assertEqual(config.storage_type, 'local')
        self.assertIsNone(config.s3_region)
        self.assertIsNone(config.gcs_project_id)
    
    def test_s3_storage_complete(self):
        """Test S3 storage with all parameters."""
        args = self.parser.parse_args([
            '--base-path', 's3://test-bucket/path',
            '--s3-region', 'us-west-2',
            '--s3-access-key-id', 'test-key',
            '--s3-secret-access-key', 'test-secret',
            '--s3-session-token', 'test-token',
            '--s3-endpoint-url', 'https://custom-endpoint'
        ])
        config = StorageConfig.from_args(args)
        
        self.assertEqual(config.base_path, 's3://test-bucket/path')
        self.assertEqual(config.storage_type, 's3')
        self.assertEqual(config.s3_region, 'us-west-2')
        self.assertEqual(config.s3_access_key_id, 'test-key')
        self.assertEqual(config.s3_secret_access_key, 'test-secret')
        self.assertEqual(config.extra_config['s3_session_token'], 'test-token')
        self.assertEqual(config.extra_config['s3_endpoint_url'], 'https://custom-endpoint')
    
    def test_gcs_storage_complete(self):
        """Test GCS storage with all parameters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"type": "service_account"}')
            creds_path = f.name
        
        try:
            args = self.parser.parse_args([
                '--base-path', 'gs://test-bucket/path',
                '--gcs-project-id', 'test-project',
                '--gcs-credentials', creds_path
            ])
            config = StorageConfig.from_args(args)
            
            self.assertEqual(config.base_path, 'gs://test-bucket/path')
            self.assertEqual(config.storage_type, 'gs')
            self.assertEqual(config.gcs_project_id, 'test-project')
            self.assertEqual(config.gcs_credentials, creds_path)
        finally:
            os.unlink(creds_path)
    
    def test_azure_storage_complete(self):
        """Test Azure storage with connection string."""
        args = self.parser.parse_args([
            '--base-path', 'azure://container/path',
            '--azure-connection-string', 'DefaultEndpointsProtocol=https;AccountName=test;'
        ])
        config = StorageConfig.from_args(args)
        
        self.assertEqual(config.base_path, 'azure://container/path')
        self.assertEqual(config.storage_type, 'azure')
        self.assertEqual(config.azure_storage_connection_string, 
                        'DefaultEndpointsProtocol=https;AccountName=test;')
    
    def test_conflicting_storage_parameters(self):
        """Test that conflicting storage parameters raise an error."""
        args = self.parser.parse_args([
            '--base-path', '/tmp/test',
            '--s3-region', 'us-west-2',
            '--gcs-project-id', 'test-project'
        ])
        
        with self.assertRaises(ValueError) as cm:
            StorageConfig.from_args(args)
        
        self.assertIn("Conflicting storage parameters", str(cm.exception))
        self.assertIn("s3", str(cm.exception))
        self.assertIn("gcs", str(cm.exception))
    
    def test_local_path_with_cloud_parameters(self):
        """Test that local path with cloud parameters raises an error."""
        args = self.parser.parse_args([
            '--base-path', '/tmp/test',
            '--s3-region', 'us-west-2'
        ])
        
        with self.assertRaises(ValueError) as cm:
            StorageConfig.from_args(args)
        
        self.assertIn("Local path provided but s3 parameters specified", str(cm.exception))
    
    def test_s3_url_with_gcs_parameters(self):
        """Test that S3 URL with GCS parameters raises an error."""
        args = self.parser.parse_args([
            '--base-path', 's3://bucket/path',
            '--gcs-project-id', 'test-project'
        ])
        
        with self.assertRaises(ValueError) as cm:
            StorageConfig.from_args(args)
        
        self.assertIn("S3 URL provided but gcs parameters specified", str(cm.exception))
    
    def test_missing_required_base_path(self):
        """Test that missing base-path raises an error."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])
    
    def test_empty_parameters_ignored(self):
        """Test that empty/None parameters are ignored."""
        args = self.parser.parse_args([
            '--base-path', 's3://bucket/path',
            '--s3-region', 'us-west-2'
            # Note: other S3 params are None/empty
        ])
        config = StorageConfig.from_args(args)
        
        self.assertEqual(config.s3_region, 'us-west-2')
        self.assertIsNone(config.s3_access_key_id)
        self.assertIsNone(config.s3_secret_access_key)
        self.assertEqual(config.extra_config, {})


class TestStorageConfigCLIIntegration(unittest.TestCase):
    """Integration tests for CLI parameter parsing with actual scripts."""
    
    def test_mock_script_usage(self):
        """Test how a script would use the new CLI functionality."""
        
        # Simulate a script that uses StorageConfig
        def mock_script_main(argv):
            parser = argparse.ArgumentParser(description="Mock script")
            parser.add_argument("--action", choices=["test"], default="test")
            
            # Add storage args using the new method
            StorageConfig.add_storage_args(parser)
            
            # Parse args
            args = parser.parse_args(argv)
            
            # Create storage config using the new method
            storage_config = StorageConfig.from_args(args)
            
            return storage_config
        
        # Test local storage
        config = mock_script_main([
            '--base-path', '/tmp/test',
            '--action', 'test'
        ])
        self.assertEqual(config.storage_type, 'local')
        
        # Test S3 storage
        config = mock_script_main([
            '--base-path', 's3://bucket/path',
            '--s3-region', 'us-east-1',
            '--action', 'test'
        ])
        self.assertEqual(config.storage_type, 's3')
        self.assertEqual(config.s3_region, 'us-east-1')
    
    def test_help_message_generation(self):
        """Test that help messages are properly generated."""
        parser = argparse.ArgumentParser()
        StorageConfig.add_storage_args(parser)
        
        # This would normally print help, but we just want to make sure it doesn't crash
        with patch('sys.stdout'):
            with self.assertRaises(SystemExit):
                parser.parse_args(['--help'])
    
    @patch('pocaduck.storage_config.StorageConfig.__post_init__')
    def test_validation_bypass_for_testing(self, mock_post_init):
        """Test that we can bypass validation for unit testing if needed."""
        # Sometimes in tests you want to create invalid configs to test error handling
        mock_post_init.return_value = None
        
        # This would normally fail validation, but with mocking it works
        config = StorageConfig(base_path="invalid://path")
        self.assertEqual(config.base_path, "invalid://path")


if __name__ == '__main__':
    unittest.main()
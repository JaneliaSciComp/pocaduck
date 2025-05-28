"""
Storage configuration for PoCADuck.

This module provides a configuration class for specifying storage details,
including local and cloud storage options (S3, GCS, Azure).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse
import argparse


@dataclass
class StorageConfig:
    """
    Configuration for storage backend (local, S3, GCS, Azure).
    
    This class handles configuration for different storage backends and provides 
    necessary parameters for DuckDB and Arrow/Parquet to access these backends.
    
    Attributes:
        base_path: Base path for storage (can be local or cloud URL)
        s3_region: AWS region for S3 storage
        s3_access_key_id: AWS access key ID for S3 storage
        s3_secret_access_key: AWS secret access key for S3 storage
        gcs_project_id: Google Cloud project ID for GCS storage
        gcs_credentials: Google Cloud credentials for GCS storage
        azure_storage_connection_string: Azure storage connection string
        extra_config: Additional configuration parameters for storage
    """
    base_path: str
    s3_region: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    gcs_project_id: Optional[str] = None
    gcs_credentials: Optional[Union[str, Dict[str, Any]]] = None
    azure_storage_connection_string: Optional[str] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process storage configuration after initialization."""
        # Parse the base path to determine storage type
        parsed_url = urlparse(self.base_path)
        self.storage_type = parsed_url.scheme if parsed_url.scheme else "local"
        
        # Validate configuration based on storage type
        if self.storage_type == "s3":
            self._validate_s3_config()
        elif self.storage_type == "gs" or self.storage_type == "gcs":
            self._validate_gcs_config()
        elif self.storage_type == "azure" or self.storage_type == "az":
            self._validate_azure_config()
    
    def _validate_s3_config(self):
        """Validate S3 configuration."""
        if not self.s3_region:
            raise ValueError("s3_region is required for S3 storage")
    
    def _validate_gcs_config(self):
        """Validate GCS configuration."""
        if not self.gcs_project_id and "GOOGLE_CLOUD_PROJECT" not in self.extra_config:
            raise ValueError("gcs_project_id is required for GCS storage")
    
    def _validate_azure_config(self):
        """Validate Azure configuration."""
        if not self.azure_storage_connection_string:
            raise ValueError("azure_storage_connection_string is required for Azure storage")
    
    def get_duckdb_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for DuckDB.
        
        Returns:
            Dict with storage-specific configuration for DuckDB.
        """
        config = {}
        
        if self.storage_type == "s3":
            if self.s3_region:
                config["s3_region"] = self.s3_region
            if self.s3_access_key_id:
                config["s3_access_key_id"] = self.s3_access_key_id
            if self.s3_secret_access_key:
                config["s3_secret_access_key"] = self.s3_secret_access_key
        
        elif self.storage_type in ("gs", "gcs"):
            if self.gcs_project_id:
                config["gcs_project_id"] = self.gcs_project_id
            if self.gcs_credentials:
                if isinstance(self.gcs_credentials, str):
                    config["gcs_key_path"] = self.gcs_credentials
                else:
                    config["gcs_credentials"] = self.gcs_credentials
        
        elif self.storage_type in ("azure", "az"):
            if self.azure_storage_connection_string:
                config["azure_storage_connection_string"] = self.azure_storage_connection_string
        
        # Add any extra configuration parameters
        config.update(self.extra_config)
        
        return config
    
    @classmethod
    def add_storage_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add storage-related arguments to an ArgumentParser.
        
        Args:
            parser: ArgumentParser to add storage arguments to
        """
        parser.add_argument("--base-path", type=str, required=True,
                            help="Base path for storage (local path, s3://, gs://, or azure://)")
        
        # S3 arguments
        s3_group = parser.add_argument_group("S3 Storage Options")
        s3_group.add_argument("--s3-region", type=str,
                             help="AWS S3 region")
        s3_group.add_argument("--s3-access-key-id", type=str,
                             help="AWS S3 access key ID")
        s3_group.add_argument("--s3-secret-access-key", type=str,
                             help="AWS S3 secret access key")
        s3_group.add_argument("--s3-session-token", type=str,
                             help="AWS S3 session token")
        s3_group.add_argument("--s3-endpoint-url", type=str,
                             help="Custom S3 endpoint URL")
        
        # GCS arguments
        gcs_group = parser.add_argument_group("Google Cloud Storage Options")
        gcs_group.add_argument("--gcs-project-id", type=str,
                              help="Google Cloud project ID")
        gcs_group.add_argument("--gcs-credentials", type=str,
                              help="Path to Google Cloud credentials JSON file")
        
        # Azure arguments
        azure_group = parser.add_argument_group("Azure Storage Options")
        azure_group.add_argument("--azure-connection-string", type=str,
                                help="Azure storage connection string")
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "StorageConfig":
        """
        Create a StorageConfig instance from parsed command line arguments.
        
        Args:
            args: Parsed arguments from ArgumentParser
            
        Returns:
            StorageConfig instance
            
        Raises:
            ValueError: If invalid parameter combinations are provided
        """
        # Start with base path
        kwargs = {"base_path": args.base_path}
        
        # Determine expected storage type from base_path
        parsed_url = urlparse(args.base_path)
        expected_storage_type = parsed_url.scheme if parsed_url.scheme else "local"
        
        # Collect provided parameters by storage type
        s3_params = {}
        gcs_params = {}
        azure_params = {}
        extra_params = {}
        
        # S3 parameters
        if hasattr(args, 's3_region') and args.s3_region:
            s3_params['s3_region'] = args.s3_region
        if hasattr(args, 's3_access_key_id') and args.s3_access_key_id:
            s3_params['s3_access_key_id'] = args.s3_access_key_id
        if hasattr(args, 's3_secret_access_key') and args.s3_secret_access_key:
            s3_params['s3_secret_access_key'] = args.s3_secret_access_key
        if hasattr(args, 's3_session_token') and args.s3_session_token:
            extra_params['s3_session_token'] = args.s3_session_token
        if hasattr(args, 's3_endpoint_url') and args.s3_endpoint_url:
            extra_params['s3_endpoint_url'] = args.s3_endpoint_url
            
        # GCS parameters
        if hasattr(args, 'gcs_project_id') and args.gcs_project_id:
            gcs_params['gcs_project_id'] = args.gcs_project_id
        if hasattr(args, 'gcs_credentials') and args.gcs_credentials:
            gcs_params['gcs_credentials'] = args.gcs_credentials
            
        # Azure parameters
        if hasattr(args, 'azure_connection_string') and args.azure_connection_string:
            azure_params['azure_storage_connection_string'] = args.azure_connection_string
        
        # Validate parameter combinations
        provided_storage_types = []
        if s3_params:
            provided_storage_types.append("s3")
        if gcs_params:
            provided_storage_types.append("gcs")
        if azure_params:
            provided_storage_types.append("azure")
        
        # Check for conflicting storage parameters
        if len(provided_storage_types) > 1:
            raise ValueError(f"Conflicting storage parameters provided for: {', '.join(provided_storage_types)}. "
                           f"Please provide parameters for only one storage type.")
        
        # Check if provided parameters match expected storage type
        if provided_storage_types:
            provided_type = provided_storage_types[0]
            if expected_storage_type == "local" and provided_type:
                raise ValueError(f"Local path provided but {provided_type} parameters specified. "
                               f"Use a {provided_type}:// URL or remove {provided_type} parameters.")
            elif expected_storage_type == "s3" and provided_type != "s3":
                raise ValueError(f"S3 URL provided but {provided_type} parameters specified.")
            elif expected_storage_type in ("gs", "gcs") and provided_type != "gcs":
                raise ValueError(f"GCS URL provided but {provided_type} parameters specified.")
            elif expected_storage_type in ("azure", "az") and provided_type != "azure":
                raise ValueError(f"Azure URL provided but {provided_type} parameters specified.")
        
        # Add the appropriate parameters
        kwargs.update(s3_params)
        kwargs.update(gcs_params)
        kwargs.update(azure_params)
        
        if extra_params:
            kwargs['extra_config'] = extra_params
        
        return cls(**kwargs)
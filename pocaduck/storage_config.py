"""
Storage configuration for PoCADuck.

This module provides a configuration class for specifying storage details,
including local and cloud storage options (S3, GCS, Azure).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse


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
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
    Configuration for storage backend (local, S3, GCS, Azure, VastDB).
    
    This class handles configuration for different storage backends and provides 
    necessary parameters for DuckDB and Arrow/Parquet to access these backends,
    or for VastDB direct access.
    
    Attributes:
        base_path: Base path for storage (can be local or cloud URL) - ignored for VastDB
        storage_type: Explicit storage type override ('local', 's3', 'gcs', 'azure', 'vastdb')
        
        # File-based storage parameters (parquet + DuckDB)
        s3_region: AWS region for S3 storage
        s3_access_key_id: AWS access key ID for S3 storage
        s3_secret_access_key: AWS secret access key for S3 storage
        gcs_project_id: Google Cloud project ID for GCS storage
        gcs_credentials: Google Cloud credentials for GCS storage
        azure_storage_connection_string: Azure storage connection string
        
        # VastDB parameters
        vastdb_endpoint: VastDB cluster endpoint URL
        vastdb_access_key: VastDB access key (S3-compatible)
        vastdb_secret_key: VastDB secret key (S3-compatible)
        vastdb_bucket: VastDB bucket name
        vastdb_schema: VastDB schema name
        vastdb_table: VastDB table name for point clouds
        
        extra_config: Additional configuration parameters for storage
    """
    base_path: Optional[str] = None
    storage_type: Optional[str] = None  # Explicit storage type override
    
    # File-based storage parameters (parquet + DuckDB)
    s3_region: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    gcs_project_id: Optional[str] = None
    gcs_credentials: Optional[Union[str, Dict[str, Any]]] = None
    azure_storage_connection_string: Optional[str] = None
    
    # VastDB parameters
    vastdb_endpoint: Optional[str] = None
    vastdb_access_key: Optional[str] = None
    vastdb_secret_key: Optional[str] = None
    vastdb_bucket: Optional[str] = None
    vastdb_schema: Optional[str] = None
    vastdb_table: str = "point_clouds"
    
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process storage configuration after initialization."""
        # Determine storage type
        if self.storage_type is None:
            # Auto-detect from base_path if not explicitly set
            if self.base_path is None:
                raise ValueError("Either storage_type must be specified or base_path must be provided")
            parsed_url = urlparse(self.base_path)
            self.storage_type = parsed_url.scheme if parsed_url.scheme else "local"
        
        # Validate configuration based on storage type
        if self.storage_type == "vastdb":
            self._validate_vastdb_config()
        elif self.storage_type == "s3":
            self._validate_s3_config()
        elif self.storage_type == "gs" or self.storage_type == "gcs":
            self._validate_gcs_config()
        elif self.storage_type == "azure" or self.storage_type == "az":
            self._validate_azure_config()
        elif self.storage_type == "local":
            if self.base_path is None:
                raise ValueError("base_path is required for local storage")
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _validate_vastdb_config(self):
        """Validate VastDB configuration."""
        required_params = [
            ('vastdb_endpoint', self.vastdb_endpoint),
            ('vastdb_access_key', self.vastdb_access_key),
            ('vastdb_secret_key', self.vastdb_secret_key),
            ('vastdb_bucket', self.vastdb_bucket),
            ('vastdb_schema', self.vastdb_schema)
        ]
        
        for param_name, param_value in required_params:
            if not param_value:
                raise ValueError(f"{param_name} is required for VastDB storage")
    
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
        # General storage arguments
        parser.add_argument("--storage-type", type=str, 
                           choices=["local", "s3", "gcs", "azure", "vastdb"],
                           help="Storage backend type (auto-detected from base-path if not specified)")
        parser.add_argument("--base-path", type=str,
                            help="Base path for storage (local path, s3://, gs://, or azure://) - not used for VastDB")
        
        # VastDB arguments
        vastdb_group = parser.add_argument_group("VastDB Storage Options")
        vastdb_group.add_argument("--vastdb-endpoint", type=str,
                                 help="VastDB cluster endpoint URL")
        vastdb_group.add_argument("--vastdb-access-key", type=str,
                                 help="VastDB access key")
        vastdb_group.add_argument("--vastdb-secret-key", type=str,
                                 help="VastDB secret key")
        vastdb_group.add_argument("--vastdb-bucket", type=str,
                                 help="VastDB bucket name")
        vastdb_group.add_argument("--vastdb-schema", type=str,
                                 help="VastDB schema name")
        vastdb_group.add_argument("--vastdb-table", type=str, default="point_clouds",
                                 help="VastDB table name (default: point_clouds)")
        
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
        kwargs = {}
        
        # Set explicit storage type if provided
        if hasattr(args, 'storage_type') and args.storage_type:
            kwargs['storage_type'] = args.storage_type
        
        # Set base path if provided (not used for VastDB)
        if hasattr(args, 'base_path') and args.base_path:
            kwargs['base_path'] = args.base_path
        
        # Collect VastDB parameters
        vastdb_params = {}
        if hasattr(args, 'vastdb_endpoint') and args.vastdb_endpoint:
            vastdb_params['vastdb_endpoint'] = args.vastdb_endpoint
        if hasattr(args, 'vastdb_access_key') and args.vastdb_access_key:
            vastdb_params['vastdb_access_key'] = args.vastdb_access_key
        if hasattr(args, 'vastdb_secret_key') and args.vastdb_secret_key:
            vastdb_params['vastdb_secret_key'] = args.vastdb_secret_key
        if hasattr(args, 'vastdb_bucket') and args.vastdb_bucket:
            vastdb_params['vastdb_bucket'] = args.vastdb_bucket
        if hasattr(args, 'vastdb_schema') and args.vastdb_schema:
            vastdb_params['vastdb_schema'] = args.vastdb_schema
        if hasattr(args, 'vastdb_table') and args.vastdb_table:
            vastdb_params['vastdb_table'] = args.vastdb_table
        
        # Collect S3 parameters
        s3_params = {}
        if hasattr(args, 's3_region') and args.s3_region:
            s3_params['s3_region'] = args.s3_region
        if hasattr(args, 's3_access_key_id') and args.s3_access_key_id:
            s3_params['s3_access_key_id'] = args.s3_access_key_id
        if hasattr(args, 's3_secret_access_key') and args.s3_secret_access_key:
            s3_params['s3_secret_access_key'] = args.s3_secret_access_key
        
        # Collect GCS parameters
        gcs_params = {}
        if hasattr(args, 'gcs_project_id') and args.gcs_project_id:
            gcs_params['gcs_project_id'] = args.gcs_project_id
        if hasattr(args, 'gcs_credentials') and args.gcs_credentials:
            gcs_params['gcs_credentials'] = args.gcs_credentials
        
        # Collect Azure parameters
        azure_params = {}
        if hasattr(args, 'azure_connection_string') and args.azure_connection_string:
            azure_params['azure_storage_connection_string'] = args.azure_connection_string
        
        # Collect extra parameters
        extra_params = {}
        if hasattr(args, 's3_session_token') and args.s3_session_token:
            extra_params['s3_session_token'] = args.s3_session_token
        if hasattr(args, 's3_endpoint_url') and args.s3_endpoint_url:
            extra_params['s3_endpoint_url'] = args.s3_endpoint_url
        
        # Check for conflicting storage parameters
        provided_storage_types = []
        if vastdb_params:
            provided_storage_types.append("vastdb")
        if s3_params:
            provided_storage_types.append("s3")
        if gcs_params:
            provided_storage_types.append("gcs")
        if azure_params:
            provided_storage_types.append("azure")
        
        if len(provided_storage_types) > 1:
            raise ValueError(f"Conflicting storage parameters provided for: {', '.join(provided_storage_types)}. "
                           f"Please provide parameters for only one storage type.")
        
        # If storage type is explicitly set, validate it matches provided parameters
        if 'storage_type' in kwargs:
            explicit_type = kwargs['storage_type']
            if provided_storage_types and explicit_type not in provided_storage_types:
                raise ValueError(f"Explicit storage type '{explicit_type}' conflicts with "
                               f"provided parameters for: {', '.join(provided_storage_types)}")
        
        # Add the appropriate parameters
        kwargs.update(vastdb_params)
        kwargs.update(s3_params)
        kwargs.update(gcs_params)
        kwargs.update(azure_params)
        
        if extra_params:
            kwargs['extra_config'] = extra_params
        
        return cls(**kwargs)
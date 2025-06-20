# VastDB Backend Design for PoCADuck

## Architecture Overview

### File-based Architecture (Parquet + DuckDB):
```
User Query → DuckDB Index → Parquet Files → Point Cloud Data
              ↑              ↑
         Complex indexing   File management
         Optimization       Worker sharding
         pipeline          
```

### New Architecture (VastDB):
```
User Query → VastDB SQL → Point Cloud Data
                ↑
         Simple key-value lookup
         No optimization needed
```

## Implementation Plan

### 1. StorageConfig Modifications

Add VastDB support to `storage_config.py`:

```python
class StorageConfig:
    def __init__(
        self, 
        storage_type: str = "vastdb",  # New default
        
        # VastDB parameters
        vastdb_endpoint: Optional[str] = None,
        vastdb_access_key: Optional[str] = None,
        vastdb_secret_key: Optional[str] = None,
        vastdb_bucket: Optional[str] = None,
        vastdb_schema: Optional[str] = None,
        vastdb_table: str = "point_clouds",
        
        # File-based parquet support
        base_path: Optional[str] = None,
        **kwargs
    ):
```

### 2. VastDB Query Implementation

New `VastDBQuery` class in `query.py`:

```python
import vastdb
import pyarrow as pa

class VastDBQuery:
    def __init__(self, storage_config: StorageConfig, **kwargs):
        self.storage_config = storage_config
        self.session = vastdb.connect(
            endpoint=storage_config.vastdb_endpoint,
            access=storage_config.vastdb_access_key,
            secret=storage_config.vastdb_secret_key
        )
        self.bucket = self.session.bucket(storage_config.vastdb_bucket)
        self.schema = self.bucket.schema(storage_config.vastdb_schema)
        self.table = self.schema.table(storage_config.vastdb_table)
        
    def get_labels(self) -> np.ndarray:
        """Get all available labels."""
        result = self.table.select(columns=['label'])
        return result['label'].to_numpy()
    
    def get_point_count(self, label: int, timing: bool = False):
        """Get point count for a label."""
        if not timing:
            result = self.table.select(
                columns=['point_count'],
                predicate=(_.label == label)
            )
            return result['point_count'][0] if len(result) > 0 else 0
        
        # Timing version
        start_time = time.time()
        result = self.table.select(
            columns=['point_count'],
            predicate=(_.label == label)
        )
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'query_type': 'vastdb_point_count',
            'sql_query': f"SELECT point_count FROM point_clouds WHERE label = {label}"
        }
        count = result['point_count'][0] if len(result) > 0 else 0
        return count, timing_info
    
    def get_points(self, label: int, use_cache: bool = True, timing: bool = False):
        """Get all points for a label."""
        if not timing:
            result = self.table.select(
                columns=['points'],
                predicate=(_.label == label)
            )
            if len(result) == 0:
                return np.array([], dtype=np.int64).reshape(0, 3)
            points_list = result['points'][0]  # Get first (and only) row
            return np.array(points_list, dtype=np.int64)
        
        # Timing version
        start_time = time.time()
        query_start = time.time()
        result = self.table.select(
            columns=['points'],
            predicate=(_.label == label)
        )
        query_time = time.time() - query_start
        
        processing_start = time.time()
        if len(result) == 0:
            points = np.array([], dtype=np.int64).reshape(0, 3)
        else:
            points_list = result['points'][0]
            points = np.array(points_list, dtype=np.int64)
        processing_time = time.time() - processing_start
        
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'query_time': query_time,
            'processing_time': processing_time,
            'query_type': 'vastdb_points',
            'sql_query': f"SELECT points FROM point_clouds WHERE label = {label}",
            'backend': 'vastdb',
            'points_returned': len(points),
            'files_accessed': 1,  # Always 1 for VastDB
            'cache_hit': False   # TODO: Implement caching if needed
        }
        
        return points, timing_info
    
    def get_blocks_for_label(self, label: int, timing: bool = False):
        """Get blocks for label (not applicable for VastDB)."""
        if not timing:
            return []  # No concept of blocks in VastDB
        
        timing_info = {
            'label': label,
            'total_time': 0.0,
            'query_type': 'vastdb_blocks',
            'blocks_found': 0,
            'note': 'Block concept not applicable for VastDB backend'
        }
        return [], timing_info
```

### 3. Unified Query Class

Modify the main `Query` class to support both backends:

```python
class Query:
    def __init__(self, storage_config: StorageConfig, **kwargs):
        self.storage_config = storage_config
        
        if storage_config.storage_type == "vastdb":
            self._backend = VastDBQuery(storage_config, **kwargs)
        elif storage_config.storage_type in ["local", "s3", "gcs", "azure"]:
            self._backend = ParquetDuckDBQuery(storage_config, **kwargs)  # File-based
        else:
            raise ValueError(f"Unsupported storage type: {storage_config.storage_type}")
    
    # Delegate all methods to the backend
    def get_labels(self) -> np.ndarray:
        return self._backend.get_labels()
    
    def get_point_count(self, label: int, timing: bool = False):
        return self._backend.get_point_count(label, timing)
    
    def get_points(self, label: int, use_cache: bool = True, timing: bool = False):
        return self._backend.get_points(label, use_cache, timing)
    
    def get_blocks_for_label(self, label: int, timing: bool = False):
        return self._backend.get_blocks_for_label(label, timing)
    
    def close(self):
        if hasattr(self._backend, 'close'):
            self._backend.close()
```

## Benefits of VastDB Backend

1. **Massive Simplification**: No file management, no optimization pipeline
2. **Better Performance**: Direct key-value access, no 500-file scattering
3. **Scalability**: Handle 500M+ point clouds without metadata overhead
4. **Simpler Timing**: Much cleaner metrics (no file distribution analysis needed)
5. **Easier Deployment**: No worker coordination, no consolidation steps

## Migration Path

1. **Phase 1**: Implement VastDB backend alongside existing parquet backend
2. **Phase 2**: Add migration utilities to move from parquet to VastDB
3. **Phase 3**: Deprecate parquet backend, make VastDB the default

## Usage Example

```python
from pocaduck import StorageConfig, Query

# VastDB configuration
config = StorageConfig(
    storage_type="vastdb",
    vastdb_endpoint="http://vip-pool.v123-xy.VastENG.lab",
    vastdb_access_key="your-access-key",
    vastdb_secret_key="your-secret-key",
    vastdb_bucket="neuron-data",
    vastdb_schema="point_clouds"
)

# Same API as before!
query = Query(config)
points = query.get_points(label=720575940611644529)
print(f"Retrieved {len(points)} points")  # Should be much faster!

# Timing analysis
points, timing = query.get_points(label=720575940611644529, timing=True)
Query.print_timing_info(timing)
# Expected: <1 second, 1 "file" accessed, >100K points/sec
```
"""
PoCADuck: A library for efficiently storing and retrieving point clouds.

PoCADuck provides an efficient way to store vast numbers of point clouds indexed 
by a uint64 label, handling both ingestion and retrieval across distributed blocks.
"""

from .storage_config import StorageConfig
from .ingestor import Ingestor
from .query import Query

__all__ = ['StorageConfig', 'Ingestor', 'Query']
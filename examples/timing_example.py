#!/usr/bin/env python
"""
Example demonstrating the timing functionality in PoCADuck Query class.

This example shows how to use the timing parameter to analyze query performance
and identify potential bottlenecks in point cloud retrieval.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocaduck.storage_config import StorageConfig
from pocaduck.query import Query

def analyze_query_performance(base_path: str, labels_to_test: list):
    """
    Analyze query performance for specified labels using timing information.
    
    Args:
        base_path: Path to the PoCADuck data directory
        labels_to_test: List of label IDs to analyze
    """
    # Create storage configuration
    storage_config = StorageConfig(base_path=base_path)
    
    # Create Query instance
    query = Query(storage_config)
    
    print(f"Data source: {base_path}")
    print(f"Using optimized data: {query.using_optimized_data}")
    print(f"Index path: {query.index_path}")
    print(f"Threads configured: {query.threads}")
    print("=" * 80)
    
    # Analyze each label
    for label in labels_to_test:
        print(f"\n🧠 ANALYZING LABEL: {label}")
        
        # Get points with timing information
        try:
            points, timing_info = query.get_points(label, timing=True)
            
            # Use the built-in pretty printer
            Query.print_timing_info(timing_info)
            
        except Exception as e:
            print(f"❌ Error querying label {label}: {str(e)}")
        
        print("-" * 80)
    
    query.close()

def main():
    """Main function to run the timing analysis example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze PoCADuck query performance with timing")
    parser.add_argument("--base-path", required=True, help="Path to PoCADuck data directory")
    parser.add_argument("--labels", required=True, help="Comma-separated list of labels to test")
    
    args = parser.parse_args()
    
    # Parse labels
    try:
        labels = [int(label.strip()) for label in args.labels.split(",")]
    except ValueError:
        print("Error: Labels must be comma-separated integers")
        return 1
    
    # Run analysis
    try:
        analyze_query_performance(args.base_path, labels)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
Test runner for PoCADuck.

Run this script to execute all tests in the tests directory.
"""

import unittest
import sys
import os

if __name__ == "__main__":
    # Add the parent directory to the path so we can import pocaduck
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    tests = loader.discover('tests')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(tests)
    
    # Exit with non-zero code if any tests failed
    sys.exit(not result.wasSuccessful())
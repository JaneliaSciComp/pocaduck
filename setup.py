from setuptools import setup, find_packages

setup(
    name="pocaduck",
    version="0.1.0",
    description="Efficient storage and retrieval of point clouds using Arrow/Parquet and DuckDB",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow",
        "duckdb",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Change as appropriate
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
)
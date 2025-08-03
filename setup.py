#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

# Main requirements
install_requires = read_requirements('requirements.txt')

# Development requirements
dev_requires = [
    'pytest>=7.4.0',
    'pytest-asyncio>=0.21.0',
    'black>=23.0.0',
    'isort>=5.12.0', 
    'flake8>=6.0.0',
    'mypy>=1.4.0',
    'pre-commit>=3.3.0',
    'jupyter>=1.0.0',
    'jupyterlab>=4.0.0',
]

# Optional dependencies
extras_require = {
    'dev': dev_requires,
    'optimization': [
        'onnxoptimizer>=0.3.0',
        'accelerate>=0.20.0',
        'optimum>=1.9.0',
    ],
    'monitoring': [
        'prometheus-client>=0.17.0',
        'grafana-api>=1.0.3',
    ],
    'profiling': [
        'py-spy>=0.3.14',
        'memory-profiler>=0.61.0',
        'nvidia-ml-py>=11.495.46',
    ],
    'database': [
        'redis>=4.6.0',
        'sqlalchemy>=2.0.0',
        'psycopg2-binary>=2.9.0',
    ],
    'visualization': [
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.15.0',
        'dash>=2.11.0',
    ],
}

# All optional dependencies
extras_require['all'] = [req for reqs in extras_require.values() for req in reqs]

setup(
    name="tensorrt-llm-inference-optimization",
    version="1.0.0",
    author="TensorRT-LLM Team",
    author_email="tensorrt-llm@example.com",
    description="High-performance GPT2 inference using TensorRT-LLM with KV Cache and FlashAttention optimizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorrt-llm/inference-optimization",
    project_urls={
        "Bug Reports": "https://github.com/tensorrt-llm/inference-optimization/issues",
        "Source": "https://github.com/tensorrt-llm/inference-optimization",
        "Documentation": "https://tensorrt-llm-inference-optimization.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "benchmark.results", "logs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "tensorrt-llm-server=server.api:main",
            "tensorrt-llm-build=scripts.build_engine:main",
            "tensorrt-llm-benchmark=scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tensorrt_llm_optimization": [
            "config/*.yaml",
            "docker/*",
            "deploy/*.yaml",
            "scripts/*.sh",
        ],
    },
    zip_safe=False,
    keywords=[
        "tensorrt",
        "llm",
        "inference",
        "optimization",
        "gpt2",
        "transformer",
        "nvidia",
        "cuda",
        "kv-cache",
        "flash-attention",
        "fastapi",
        "onnx",
    ],
    platforms=["Linux", "Windows", "macOS"],
    license="Apache License 2.0",
    
    # Metadata for PyPI
    maintainer="TensorRT-LLM Team",
    maintainer_email="tensorrt-llm@example.com",
    
    # Additional metadata
    project_name="TensorRT-LLM Inference Optimization",
    
    # Command line interface
    scripts=[
        "scripts/setup_env.sh",
        "scripts/build_engine.sh", 
        "scripts/run_server.sh",
        "scripts/benchmark.sh",
    ],
    
    # Configuration for different installation scenarios
    cmdclass={},
    
    # Dependency links for development versions
    dependency_links=[],
    
    # Test suite
    test_suite="tests",
    tests_require=dev_requires,
    
    # Options for different package managers
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal (requires specific Python versions)
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# Post-installation message
def print_post_install_message():
    print("""
    🚀 TensorRT-LLM Inference Optimization installed successfully!
    
    Next steps:
    1. Set up the environment: ./scripts/setup_env.sh
    2. Build TensorRT engine: ./scripts/build_engine.sh
    3. Start the server: ./scripts/run_server.sh
    4. Run benchmarks: ./scripts/benchmark.sh
    
    For detailed documentation, visit:
    https://tensorrt-llm-inference-optimization.readthedocs.io/
    
    For support and issues:
    https://github.com/tensorrt-llm/inference-optimization/issues
    """)

if __name__ == "__main__":
    print_post_install_message()
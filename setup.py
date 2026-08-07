"""
ELA Analysis Package Setup

Energy Landscape Analysis (ELA) パッケージのセットアップファイル
OSS対応のため、柔軟な依存関係管理とインストールオプションを提供します。
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# パッケージ情報
PACKAGE_NAME = "ela-analysis"
VERSION = "1.0.0"
DESCRIPTION = "Energy Landscape Analysis for Psychological Questionnaires"
AUTHOR = "ELA Analysis Team"
AUTHOR_EMAIL = "masa10223@nagoya-u.jp"
URL = "https://github.com/example/ela-analysis"
LICENSE = "MIT"

# 長い説明の読み込み
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 基本依存関係
BASE_REQUIREMENTS = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scipy>=1.9.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "networkx>=2.8.0",
    "tqdm>=4.64.0",
    "openpyxl>=3.0.0",
    "PyYAML>=6.0",
]

# オプション依存関係
EXTRA_REQUIREMENTS = {
    "cuda": [
        "cupy-cuda11x>=10.0.0",  # CUDA 11.x用
        # "cupy-cuda12x>=12.0.0",  # CUDA 12.x用（必要に応じてコメントアウト）
    ],
    "visualization": [
        "plotly>=5.0.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
    ],
    "testing": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    "dev": [
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

# 全てのオプション依存関係を統合
EXTRA_REQUIREMENTS["all"] = [
    req for reqs in EXTRA_REQUIREMENTS.values() for req in reqs
]

# パッケージの検出
packages = find_packages(where=".", exclude=["tests*", "docs*", "examples*"])

# パッケージデータ
package_data = {
    "ela_analysis": [
        "config/*.json",
        "config/*.yaml",
        "data/*.csv",
        "data/*.xlsx",
    ]
}

# エントリーポイント
entry_points = {
    "console_scripts": [
        "ela-pipeline=ela_analysis.src.ela_pipeline:main",
        "ela-config=ela_analysis.src.config:create_example_config_file",
    ],
}

# セットアップ
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=packages,
    package_data=package_data,
    entry_points=entry_points,
    python_requires=">=3.8",
    install_requires=BASE_REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=[
        "energy landscape",
        "psychological analysis",
        "questionnaire",
        "data analysis",
        "visualization",
        "network analysis",
    ],
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}/docs",
    },
    include_package_data=True,
    zip_safe=False,
)

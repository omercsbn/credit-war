"""
Package configuration for CREDIT WAR
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="credit-war",
    version="1.0.0",
    author="Ömercan Sabun",
    author_email="omercansabun@icloud.com",
    description="A deterministic multi-agent strategic environment for financial risk modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omercsbn/credit-war",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "creditwar=credit_war.cli:main",
        ],
    },
)

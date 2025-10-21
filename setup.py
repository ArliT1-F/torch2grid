from setuptools import setup, find_packages
import os


def read_readme():
    with open("README.md", 'r', encoding='utf-8') as fh:
        return fh.read()
    

def read_requirements():
    with open("torch2grid/requirements.txt", 'r', encoding='utf-8') as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith('#')]
    


setup(
    name='torch2grid',
    version='1.0.0',
    author='torch2grid Devs/Contributors',
    author_email="arliturka@gmail.com",
    description="A lightweight Python tool for visualizing PyTorch model weights and architectures",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ArliT1-F/torch2grid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",

    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
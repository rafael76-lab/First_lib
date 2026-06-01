from setuptools import setup, find_packages

setup(
    name="RAG_I_Q_Pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-text-splitters",
        "langchain-chroma",
        "pypdf",
        "sentence-transformers"
    ],
)
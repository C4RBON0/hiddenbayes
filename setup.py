from setuptools import setup, find_packages

setup(
    name="hiddenbayes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Optional dependencies
    author="Angel Cervera Ronda",
    description="A Hidden Naive Bayes library, it will give support to scikit-learn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/C4RBON0/hiddenbayes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

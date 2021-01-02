from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="capi",
    version="0.1.0",
    author="Samuel Sokota",
    license="LICENSE",
    description="Example implementation of CAPI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ssokota/capi",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy >=1.15.0",
        "matplotlib >= 2.0.0",
        "pandas >= 1.0.3",
        "seaborn >= 0.10.1",
        "torch >= 1.4.0"
    ],
)

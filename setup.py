from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="topos",
    version="0.1.0",
    author="OTNO Contributors",
    author_email="example@example.com",
    description="TOPOS: Topological Optimal-transport Partitioned Operator Solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mamtasaini003/OTNO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)

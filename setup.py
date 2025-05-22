from setuptools import find_packages, setup

setup(
    name="pils",
    version="0.0.1",
    author="Anonymous Authors",
    author_email="anonymous@example.com",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
)

from setuptools import find_packages, setup

setup(
    name="pils",
    version="0.0.1",
    author="anonymous",
    author_email="anon@myo.us",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
)

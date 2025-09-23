# setup.py
from setuptools import find_packages, setup

setup(
    name="HAVOK_ML",
    version="0.1",
    packages=find_packages(),  # havok/ と scripts/ 以下のパッケージを自動検出
)

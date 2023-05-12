"""Install the PyPARRM package."""

from setuptools import setup

setup(
    name="pyparrm",
    version="devel",
    package_dir={"": "src/"},
    packages=["pyparrm", "pyparrm._utils"],
)

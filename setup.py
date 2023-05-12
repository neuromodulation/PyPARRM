"""Install the PyPARRM package."""

from setuptools import setup

setup(
    name="PyPARRM",
    version="devel",
    package_dir={"": "src/"},
    packages=["pyparrm", "pyparrm._utils"],
)

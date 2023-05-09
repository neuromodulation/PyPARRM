"""Install the PyPARRM package."""

from setuptools import setup

setup(
    name="pyparrm",
    version="0.0.1dev",
    package_dir={"": "src/"},
    packages=["pyparrm", "pyparrm._utils"],
)

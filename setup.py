"""Install the PyPARRM package."""

from setuptools import setup

setup(
    name="PyPARRM",
    version="1.0.0",
    package_dir={"": "src/"},
    packages=["pyparrm", "pyparrm._utils"],
)

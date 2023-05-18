"""Install the PyPARRM package."""

from setuptools import setup

setup(
    name="PyPARRM",
    version="1.1.0dev",
    package_dir={"": "src"},
    packages=["pyparrm", "pyparrm._utils"],
)

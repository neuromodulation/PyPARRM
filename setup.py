"""Install the PyPARRM package."""

from setuptools import setup

setup(
    name="PyPARRM",
    version="1.0.0",
    license="MIT",
    author="Thomas Samuel Binns",
    author_email="t.s.binns@outlook.com",
    package_dir={"": "src/"},
    packages=["pyparrm", "pyparrm._utils"],
    url="https://github.com/neuromodulation/PyPARRM",
    install_requires=["numpy", "scipy", "matplotlib", "pqdm"],
)

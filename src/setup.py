import re
from pathlib import Path

from setuptools import find_packages, setup

# get version from vemreview/__init__.py
with open("vemreview/__init__.py") as f:
    infos = f.readlines()
__version__ = ""
for line in infos:
    if "__version__" in line:
        match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", line)
        __version__ = match.groups()[0]

# read the contents of the README file
# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.rst").read_text()
long_description = "Scripts to reproduce the figures of the VEM review paper"

requirements = [
    "numpy",
    "scipy",
    "h5py",
    "scikit-learn>=0.23",
    "matplotlib",
    "networkx",
    "pydot",
    "nltk",
    "pycortex>=1.2.4",
    "himalaya",
    "pymoten",
    "datalad",
    "voxelwise_tutorials",
    "torch",
]

extras_require = {
    "docs": ["sphinx", "sphinx_gallery", "numpydoc", "nbformat"],
    "github": ["pytest"],
}


if __name__ == "__main__":
    setup(
        name="vemreview",
        maintainer="Matteo Visconti di Oleggio Castello",
        maintainer_email="mvdoc@berkeley.edu",
        description="Scripts to reproduce the figures of the VEM review paper",
        license="BSD (3-clause)",
        version=__version__,
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras_require,
        long_description=long_description,
        long_description_content_type="text/x-rst",
    )

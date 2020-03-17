"""
Simple check list from AllenNLP repo:
https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1.  Change the version in __init__.py, setup.py as well as docs/source/conf.py.

2.  Commit these changes with the message: "Release: VERSION"

3.  Add a tag in git to mark the release:
        git tag VERSION -m 'Adds tag VERSION for pypi'

    Push the tag to git:
        git push --tags origin master

4.  Build both the sources and the wheel.

    Do not change anything in setup.py between creating the wheel and the
    source distribution (obviously).

    For the wheel, run: "python setup.py bdist_wheel" in the top level
    directory.  (this will build a wheel for the python version you use to
    build it).

    For the sources, run: "python setup.py sdist" You should now have a /dist
    directory with both .whl and .tar.gz source versions.

5.  Check that everything looks correct by uploading package to test server:

    twine upload dist/* -r pypitest

    (pypi suggest using twine as other methods upload files via plaintext.)

    You may have to specify the repository url,
    use the following command then:
        twine upload dist/* -r pypitest\
        --repository-url=https://test.pypi.org/legacy/

    Check that you can install it in a virtualenv by running:
        pip install -i https://testpypi.python.org/pypi transformers

6.  Upload the final version to actual pypi:

    twine upload dist/* -r pypi

7.  Copy the release notes from RELEASE.md to the tag in github.

"""
import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname: str) -> str:
    """ Read and return README as str. """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="asta",
    version="0.0.1",
    author="Brendan Whitaker",
    author_email="...",
    description=("Shape annotations for homogeneous numpy arrays and pytorch tensors."),
    license="GPLv3",
    packages=["asta"],
    long_description=read("README"),
    long_description_content_type="text",
    install_requires=["numpy", "sympy"],
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7",
    ],
)

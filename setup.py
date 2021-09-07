# setup.py

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

about = {}
with open("./autonomy/__about__.py", encoding="utf-8") as f:
    exec(f.read(), about)

install_requires = [
    "pyphi >= 1.2.0",
    "numpy >=1.11.0",
    "scipy >=0.13.3",
    "pandas >= 1.2.4",
    "networkx >= 2.5.1",
    "pyyaml >=3.13",
    "matplotlib >= 3.4.2",
]

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    license=about["__license__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    keywords=(
        "autonomy causality causal-modeling causation "
        "information-theory dynamical-system graph-theory"
        "integrated-information-theory iit integrated-information "
        "modeling"
    ),
    packages=find_packages(exclude=["docs", "test"]),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={"IIT Website": "http://integratedinformationtheory.org/",},
)

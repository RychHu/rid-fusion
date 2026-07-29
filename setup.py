from setuptools import setup, find_packages

setup(
    name="rid-fusion",
    version="0.2.0",
    description="Multi-Protocol Remote ID Signal Fusion Engine — "
                "protocol-agnostic tokenization, cross-modal attention fusion, "
                "and meta-learning adaptation for heterogeneous drone RID signals.",
    author="Kongyu Technologies",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)

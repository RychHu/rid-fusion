from setuptools import setup, find_packages

setup(
    name="rid-fusion",
    version="0.4.0",
    description="Explainable multi-source Remote ID association and state fusion studio",
    author="RychHu",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    entry_points={
        "console_scripts": [
            "rid-fusion = rid_fusion.desktop_api:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)

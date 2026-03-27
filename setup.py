from setuptools import setup, find_packages

setup(
    name="self-model",
    version="0.1.0",
    description="Self-Reification Feature Discovery in Instruction-Tuned Language Models",
    author="Brian DeCamp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "accelerate>=0.27",
        "bitsandbytes>=0.43",
        "numpy>=1.24",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "jupyter>=1.0",
    ],
)

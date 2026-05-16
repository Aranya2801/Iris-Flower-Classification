from setuptools import setup, find_packages

setup(
    name="iris-flower-classification",
    version="2.0.0",
    author="Aranya",
    description="Advanced ML pipeline for Iris flower species classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aranya2801/Iris-Flower-Classification",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "iris-train=src.train:train_all",
            "iris-predict=src.predict:predict_single",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

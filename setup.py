from setuptools import setup, find_packages

setup(
    name="bear-detection-system",
    version="1.0.0",
    description="Advanced Bear Detection and Species Classification System",
    author="Bear Detection Team",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "opencv-python>=4.6.0",
        "PyQt5>=5.15.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "pytorch-wildlife",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
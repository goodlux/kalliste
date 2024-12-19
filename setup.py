from setuptools import setup, find_packages

setup(
    name="kalliste",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'ultralytics==8.3.51',
        'opencv-python-headless',
        'pillow',
        'numpy',
    ],
)
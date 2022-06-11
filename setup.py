from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wav2letter",
    version="0.0.1",
    author="Bilal Ahmed",
    author_email="ahmedb@purdue.edu",
    description="ASR using purely convolution based network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bilalhsp/Wav2Letter/",
    packages=find_packages(),

    install_requires=[
        'numpy', 'scipy','pytorch','torchaudio' , 'pysoundfile', 'pandas', 'pytorch-lightning',
    ],
)
from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rotational_update",
    packages=[
        "rotational_update",
        "rotational_update.layers",
        "rotational_update.layers.functions"
        ],

    version="0.0.17",
    license="MIT",

    install_requires=["torch>=1.3.0",],

    author="Tetsuya Hori",
    author_email="axis38akasira@gmail.com",

    url="https://github.com/ryhoh/Rotational_update",

    description="Rotational_update module for PyTorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='rotational rotational-update',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

import setuptools

# Package meta-data.
NAME = 'gridmix'
VERSION = '0.0.1'
AUTHOR = 'Ilya Dobrynin'
EMAIL = 'iliadobrynin@yandex.ru'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description="GridMix augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch>=1.5.1",
        "numpy>=1.19.4"
    ]
)

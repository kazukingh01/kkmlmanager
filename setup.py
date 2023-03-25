from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkmlmanager*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkmlmanager',
    version='1.1.0',
    description='my object detection library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkmlmanager",
    author='kazuking',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.24.2",
        "pandas>=1.5.3",
        "scikit-learn>=1.2.2",
        "matplotlib>=3.6.3",
    ],
    python_requires='>=3.11.2'
)

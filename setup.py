from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkmlmanager*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkmlmanager',
    version='1.2.3',
    description='usefule library for ML (mainly for table data).',
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
        "pandas==2.2.1",
        "numpy==1.26.4",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.3",
        "joblib==1.3.2",
        "umap-learn==0.5.5",
        "iterative-stratification==0.1.7",
        "setuptools>=62.0.0",
        "wheel>=0.37.0",
    ],
    python_requires='>=3.12.2'
)

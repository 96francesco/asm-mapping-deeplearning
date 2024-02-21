from setuptools import find_packages, setup

setup(
    name='src',
    version='0.1.0',
    description='Deep learning models for ASM mapping from satellite images.',
    author='Francesco Pasanisi',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)

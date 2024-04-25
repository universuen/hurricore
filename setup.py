from setuptools import setup, find_packages

setup(
    name='hurricore',
    version='0.1.1',
    packages=find_packages(),
    license='MIT',
    description='A deep learning framework based on PyTorch and Accelerate.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/universuen/hurricore',
    author='universuen',
    author_email='universuen@gmail.com'
)

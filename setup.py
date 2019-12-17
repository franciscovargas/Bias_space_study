from setuptools import setup, find_packages
from codecs import open
from os import path
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

here = path.abspath(path.dirname(__file__))
install_reqs = parse_requirements(here + '/requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='L101_Bias_space_study',
    version='1.0',
    description='L101 project on Bias in word embeddings',
    long_description=long_description,
    url='https://github.com/franciscovargas/L101_Bias_space_study.git',
    author='Francisco Vargas',
    author_email='vargfran@gmail.com',
    license='MIT',
    keywords='NLP, Kernel Methods, RKHS, Linear Algebra, PCA, Kernel PCA',
    install_requires=reqs,
)

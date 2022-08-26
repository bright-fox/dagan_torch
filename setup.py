from setuptools import setup, find_packages
import pathlib

project_path = pathlib.Path(__file__).parent.resolve()
long_descr = (project_path/'README.md').read_text(encoding='utf-8')

setup(
    name='dagan_torch',
    version='0.1.0',
    author='bright-fox',
    description='Alteration of DAGAN',
    long_description=long_descr,
    url='https://github.com/bright-fox/dagan_torch',
    license='LICENSE.txt',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)

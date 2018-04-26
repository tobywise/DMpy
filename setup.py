try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='DMpy',
    version='0.1',
    packages=['DMpy'],
    url='',
    license='',
    author='Toby Wise',
    author_email='t.wise@ucl.ac.uk',
    install_requires=[
        'functools32',
        'matplotlib',
        'numpy',
        'pandas',
        'pymc3',
        'scipy',
        'seaborn',
        'Theano',
    ],
    description=''
)

"""
Setup script for CGI_py - Python implementation of Causality Graphical Inference
"""

from setuptools import setup, find_packages

setup(
    name='cgipy',
    version='1.0.0',
    description='CGI (Causality Graphical Inference) - Python Implementation',
    author='CGI Authors',
    author_email='',
    url='https://github.com/Causality-Inference/CGI',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.5.0',
    ],
    extras_require={
        'full': [
            'scikit-learn>=0.22.0',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

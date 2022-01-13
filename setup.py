#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'rasa>=2.8,<3',
    'paddlepaddle==2.2.1',
    'paddlenlp==2.2.3',
]

test_requirements = [ ]

setup(
    author="Simon Liang",
    author_email='simon@x-tech.io',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Rasa NLU Components with PaddleNLP",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rasa_paddlenlp',
    name='rasa_paddlenlp',
    packages=find_packages(include=['rasa_paddlenlp', 'rasa_paddlenlp.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/botisan-ai/rasa-paddlenlp',
    version='0.2.0',
    zip_safe=False,
)

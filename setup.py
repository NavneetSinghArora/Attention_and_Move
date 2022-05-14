from setuptools import setup, find_packages

NAME = 'Attention_and_Move'
DESCRIPTION = 'A python package for multi-agent communication,  visual-path navigation, object detection and object movement.'
AUTHOR = 'Team 1: Computer Vision Master Project'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.0.1'
LICENSE = '(c) Copyright by author'

with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', 'r') as history_file:
    history = history_file.read()

requirements = [
    "Click",
    "intake"
]

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    python_requires=REQUIRES_PYTHON,
    install_requires=requirements,
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'attention_and_move = src.scripts.execute:cli',
        ],
    },
    zip_safe=False,

    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
    ]
)

import platform
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

if platform.system() == "Darwin":
    system = "macosx"
else:
    system = "linux"

python_version = ''.join(platform.python_version_tuple()[0:2])

requirements = [
    "ai2thor==4.2.0",
    "aws-requests-auth==0.4.3",
    "botocore==1.26.4",
    "certifi==2020.6.20",
    "charset-normalizer==2.0.12",
    "click==8.1.3",
    "clip @ git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620",
    "cycler==0.11.0",
    "Flask==2.1.2",
    "fonttools==4.33.3",
    "ftfy==6.1.1",
    "idna==3.3",
    "itsdangerous==2.1.2",
    "Jinja2==3.1.2",
    "jmespath==1.0.0",
    "jproperties==2.1.1",
    "kiwisolver==1.4.2",
    "MarkupSafe==2.1.1",
    "matplotlib==3.5.2",
    "msgpack==1.0.3",
    "numpy==1.22.3",
    "opencv-python==4.5.5.64",
    "packaging==21.3",
    "Pillow==9.1.1",
    "progressbar2==4.0.0",
    "pyparsing==3.0.9",
    "python-dateutil==2.8.2",
    "python-utils==3.2.3",
    "python-xlib==0.31",
    "PyYAML==6.0",
    "regex==2022.4.24",
    "requests==2.27.1",
    "six==1.16.0",
    f"torch @ https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp{python_version}-cp{python_version}-{system}_x86_64.whl",
    f"torchaudio @ https://download.pytorch.org/whl/cu113/torchaudio-0.11.0%2Bcu113-cp{python_version}-cp{python_version}-{system}_x86_64.whl",
    f"torchvision @ https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp{python_version}-cp{python_version}-{system}_x86_64.whl",
    "tqdm==4.64.0",
    "typing_extensions==4.2.0",
    "tensorboard==2.9.1",
    "urllib3==1.26.9",
    "wcwidth==0.2.5",
    "Werkzeug==2.1.2"
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

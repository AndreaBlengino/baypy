from setuptools import find_packages, setup
import subprocess

version = subprocess.run(['git', 'describe', '--tags'], stdout = subprocess.PIPE).stdout.decode('utf-8').strip()

with open('README.md', 'r') as f:
    long_description = f.read()

def read_requirements(path: str):
    with open(path) as file:
        lines = file.readlines()

    return [line.replace('==', ' >= ').replace('\n', '') for line in lines]

basic_requirements = read_requirements(r'requirements/common.txt')
extras_dev = read_requirements(r'requirements/dev.txt')
extras_docs = read_requirements(r'requirements/docs.txt')
extras_tests = read_requirements(r'requirements/tests.txt')


setup(name = 'baypy',
      version = version,
      description = "A python package for solving bayesian regression models "
                    "through a Monte Carlo Markov chain sampling",
      packages = find_packages(where = '.'),
      py_modules = ['__init__', 'utils'],
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = 'https://github.com/AndreaBlengino/baypy',
      project_urls = {'Source': 'https://github.com/AndreaBlengino/baypy',
                      'Tracker': 'https://github.com/AndreaBlengino/baypy/issues',
                      'Documentation': 'https://baypy.readthedocs.io/en/latest/index.html'},
      author = 'Andrea Blengino',
      author_email = 'ing.andrea.blengino@gmail.com',
      license = 'GNU GPL3',
      classifiers = ['Intended Audience :: Education',
                     'Intended Audience :: Manufacturing',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                     'Operating System :: OS Independent',
                     'Programming Language :: Python',
                     'Programming Language :: Python :: 3.9',
                     'Programming Language :: Python :: 3.10',
                     'Programming Language :: Python :: 3.11',
                     'Topic :: Scientific/Engineering',
                     'Topic :: Scientific/Engineering :: Mathematics'],
      install_requires = basic_requirements,
      extras_require = {'dev': extras_dev,
                        'docs': extras_docs,
                        'tests': extras_tests},
      python_requires = '>=3.9, <3.12')

from setuptools import find_packages, setup
import subprocess

version = subprocess.run(['git', 'describe', '--tags'], stdout = subprocess.PIPE).stdout.decode('utf-8').strip()

with open('README.md', 'r') as f:
    long_description = f.read()

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
                      'Tracker': 'https://github.com/AndreaBlengino/baypy/issues'},
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
      install_requires = ['matplotlib >= 3.7.2',
                          'numpy >= 1.25.2',
                          'pandas >= 2.0.3',
                          'scipy >= 1.11.1'],
      extras_require = {'dev': ['sphinx >= 7.2.6',
                                'm2r2 >= 0.3.3.post2',
                                'furo >= 2023.8.19',
                                'tox >= 4.5.1',
                                'hypothesis >= 6.84.3'
                                'twine >= 4.0.2']},
      python_requires = '>=3.9, <3.12')

from setuptools import setup
import subprocess


version = subprocess.run(['git', 'describe', '--tags'], stdout = subprocess.PIPE).stdout.decode('utf-8').strip()


setup(version = version)

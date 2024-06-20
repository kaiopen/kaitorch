from pathlib import Path
from setuptools import setup, find_packages
import shutil


root = Path.cwd()
dir_build = root.joinpath('build')
dir_dist = root.joinpath('dist')
dir_egg_info = root.joinpath('kaitorch.egg-info')

if dir_build.is_dir():
    shutil.rmtree(dir_build)
if dir_dist.is_dir():
    shutil.rmtree(dir_dist)
if dir_egg_info.is_dir():
    shutil.rmtree(dir_egg_info)

setup(
    name='kaitorch',
    version='1.0.0',
    packages=find_packages(),
    author='kaiopen',
    author_email='kaiopen@foxmail.com'
)

if dir_build.is_dir():
    shutil.rmtree(dir_build)
if dir_dist.is_dir():
    shutil.rmtree(dir_dist)
if dir_egg_info.is_dir():
    shutil.rmtree(dir_egg_info)

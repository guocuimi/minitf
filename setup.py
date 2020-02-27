from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    lic = f.read()

with open('requirements.txt') as f:
    reqs = list(f.read().strip().split('\n'))

setup(
    name='minitf',
    version='0.2.2.1',
    description='Simplified version of Tensorflow for learning purposes.',
    long_description=readme,
    author='Michael Mi',
    author_email='guocuimi@gmail.com',
    url='https://github.com/guocuimi/minitf',
    license=lic,
    install_requires=reqs,
    packages=find_packages(exclude=('tests')),
    package_data={'': ['README.md', 'LICENSE']},
    keywords=['Deep learning', 'Tensorflow', 'Autodiff', 'Backpropagation',
              'Gradients', 'Neural networks', 'Python', 'Numpy'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License'],
)

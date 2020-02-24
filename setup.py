from setuptools import find_packages
from setuptools import setup

setup(
    name='minitf',
    version='0.2.2',
    description='Simplified version of Tensorflow for learning purposes.',
    author='Michael Mi',
    author_email='guocuimi@gmail.com',
    install_requires=['numpy>=1.9.1', 'matplotlib>=3.1.1'],
    keywords=['Deep learning', 'Tensorflow', 'Autodiff', 'Backpropagation',
              'Gradients', 'Neural networks', 'Python', 'Numpy'],
    url='https://guocuimi.github.io/minitf/',
    license='MIT',
    classifiers=['Development Status :: 1 - Planning',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7'],
    packages=find_packages(),
)

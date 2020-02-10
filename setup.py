from setuptools import setup
from setuptools import find_packages

setup(
    name='minitf',
    version='0.1',
    description='Simplified version of Tensorflow for learning purposes.',
    author='Michael Mi',
    author_email='guocuimi@gmail.com',
    install_requires=['numpy>=1.9.1'],
    keywords=['Deep learning', 'Tensorflow', 'Autodiff', 'Backpropagation',
              'Gradients', 'Neural networks', 'Python', 'Numpy'],
    url='https://github.com/guocuimi/minitf',
    license='MIT',
    classifiers=['Development Status :: 1 - Planning',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7'],
    packages=find_packages(),
)

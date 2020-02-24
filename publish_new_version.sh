#!/bin/bash

# Deleting previous distribution archives
rm -rf dist/*

# Generating distribution archives
python setup.py sdist bdist_wheel

# Uploading the distribution archives
twine upload dist/*
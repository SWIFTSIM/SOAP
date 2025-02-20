#!/bin/bash

# Check if we can run pip
# This also serves as a check for python3
python3 -m pip --version > /dev/null
if [[ $? -ne 0 ]]
then
  echo "ERROR: cannot run 'python3 -m pip'"
  exit 1
fi

# Check if the virtual environment with black exists
if [ ! -d black_formatting_env ]
then
  echo "Formatting environment not found, installing it..."
  python3 -m venv black_formatting_env
  ./black_formatting_env/bin/python3 -m pip install black
fi
# Now we know exactly which black to use
black="./black_formatting_env/bin/python3 -m black"

# Make sure we don't try and format any virtual environments
files=$(echo {compression/*.py,misc/*.py,SOAP/*.py,SOAP/*/*.py,tests/*.py})

# Run formatting (pass --check to see what changes would be made)
$black -t py38 $files


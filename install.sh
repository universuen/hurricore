#!/bin/bash

# This line ignores all errors including broken pipes
set +e

# Trap and ignore SIGPIPE
trap '' PIPE

# Check if the user provided a project name as an argument
if [ $# -eq 0 ]
then
    printf "\033[31mError: Please provide a project name as an argument.\033[0m\n" >&2
    exit 1
fi

# Assign the project name to a variable
PROJECT_NAME=$1

# Get the absolute path to the current directory
CURRENT_DIR=$(pwd)

# Define the package name
PACKAGE_NAME="hurricore"

# Check if the package is already installed and install if not
if pip list 2>/dev/null | grep -Fq $PACKAGE_NAME
then
    printf "\033[33mWarning: Package '%s' is already installed, skipping installation.\033[0m\n" "$PACKAGE_NAME"
else
    pip install -e . 2>/dev/null
    printf "\033[32mPackage '%s' installed successfully.\033[0m\n" "$PACKAGE_NAME"
fi

# Check if the project directory already exists
if [ -d "./projects/${PROJECT_NAME}" ]
then
    printf "\033[33mWarning: Project directory '%s' already exists, skipping creation.\033[0m\n" "$PROJECT_NAME"
else
    cp -r ./projects/_template "./projects/${PROJECT_NAME}"
    printf "\033[32mProject directory created: %s\033[0m\n" "$PROJECT_NAME"
fi

# Final message
printf "You can find the project at \033[34m%s/projects/%s\033[0m\n" "$CURRENT_DIR" "$PROJECT_NAME"

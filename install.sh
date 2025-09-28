#!/bin/bash

# This line ignores all errors including broken pipes
set +e

# Trap and ignore SIGPIPE
trap '' PIPE

# Check if the user provided a project name as an argument
if [ $# -eq 0 ]
then
    printf "\033[33mWarning: No project name provided, install the package only.\033[0m\n"
    PROJECT_NAME=""
else
    PROJECT_NAME=$1
fi

# Get the absolute path to the current directory
CURRENT_DIR=$(pwd)

# Define the package name
PACKAGE_NAME="hurricore"

# Check if the package is already installed and install if not
if pip3 list | grep -Fq $PACKAGE_NAME
then
    printf "\033[33mWarning: Package '%s' is already installed, skipping installation.\033[0m\n" "$PACKAGE_NAME"
else
    pip3 install -e .[projects] -i https://mirrors.tencent.com/pypi/simple/
    rm -rf ${PACKAGE_NAME}.egg-info
    printf "\033[32mPackage '%s' installed successfully.\033[0m\n" "$PACKAGE_NAME"
fi

# Only proceed with project directory creation if a project name is provided
if [ -n "$PROJECT_NAME" ]
then
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
fi

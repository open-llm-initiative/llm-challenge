#!/bin/bash

# Step 1: Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Step 3: Install dependencies from requirements.txt
echo "Installing dependencies..."
if [ $# -ne 0  && $1 == "llm"]; then
    pip install -r requirements_llm.txt
    echo "installing python dependencies to use with CUDA"
else   
    pip install -r requirements.txt
    echo "installing python dependencies to use with the web application"
fi

# Step 4: Configure environment variables for developer keys
# Load environment variables from .env file
echo "Configuring environment variables..."
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Setup complete. Your environment is ready."
else
    echo ".env file not found! Please create one with your developer keys."
fi


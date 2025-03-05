#!/bin/bash
# Setup script for the Multi-Agent Assessment Framework

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Multi-Agent Assessment Framework...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

echo -e "Detected Python version: ${YELLOW}$python_version${NC}"

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required.${NC}"
    echo -e "Please upgrade your Python installation and try again."
    exit 1
fi

# Create virtual environment
echo -e "\n${GREEN}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "\n${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "\n${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install the package in development mode
echo -e "\n${GREEN}Installing package in development mode...${NC}"
pip install -e .

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "\n${GREEN}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit the .env file with your API keys and configuration.${NC}"
fi

# Create results directory
echo -e "\n${GREEN}Creating results directory...${NC}"
mkdir -p results

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\nTo activate the virtual environment in the future, run:"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "\nTo run a benchmark:"
echo -e "${YELLOW}python src/main.py benchmark --use-case article_writing --topic \"Your Topic\"${NC}"
echo -e "\nTo analyze results:"
echo -e "${YELLOW}python src/main.py analyze --plots${NC}"
echo -e "\nOr use the command-line tool:"
echo -e "${YELLOW}maaf benchmark --use-case article_writing${NC}"

# Deactivate virtual environment
deactivate 
#!/bin/bash

# Strands Voice Chat - Installation Script
# This script installs all dependencies needed to run the voice chat tools

set -e  # Exit on error

echo "🎤 Strands Voice Chat - Installation Script"
echo "==========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10 or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"
echo ""

# Detect OS for PyAudio installation instructions
os_type=$(uname -s)
echo "Detected OS: $os_type"
echo ""

# Check if portaudio is installed (required for PyAudio)
echo "Checking for PortAudio (required for PyAudio)..."
if [ "$os_type" = "Darwin" ]; then
    # macOS
    if ! brew list portaudio &>/dev/null; then
        echo "⚠️  PortAudio not found. Installing with Homebrew..."
        if command -v brew &>/dev/null; then
            brew install portaudio
            echo "✅ PortAudio installed"
        else
            echo "❌ Error: Homebrew not found. Please install Homebrew or install PortAudio manually."
            echo "   Visit: https://brew.sh"
            exit 1
        fi
    else
        echo "✅ PortAudio already installed"
    fi
elif [ "$os_type" = "Linux" ]; then
    # Linux
    if ! dpkg -l | grep -q portaudio19-dev; then
        echo "⚠️  PortAudio not found. Please install it manually:"
        echo "   Ubuntu/Debian: sudo apt-get install portaudio19-dev"
        echo "   Fedora: sudo dnf install portaudio-devel"
        echo "   Arch: sudo pacman -S portaudio"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✅ PortAudio already installed"
    fi
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
echo "✅ pip upgraded"
echo ""

# Install the package
echo "Installing Strands Voice Chat with all dependencies..."
echo "This may take a few minutes..."
echo ""

# Check if user wants to install all providers
read -p "Install all model providers (OpenAI, Gemini)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing with all providers..."
    python3 -m pip install -e ".[all]"
else
    echo "Installing with Nova Sonic only..."
    python3 -m pip install -e .
fi

echo ""
echo "✅ Installation complete!"
echo ""


echo "🚀 Happy voice chatting!"

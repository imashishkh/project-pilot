#!/usr/bin/env python3
"""
Setup script for MacAgent.

This script configures the package for installation via pip.
"""

import os
import sys
from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Determine dependencies based on platform
dependencies = [
    "PyQt5>=5.15.9",  # UI components
    "pyautogui>=0.9.53",  # Screen control
    "pynput>=1.7.6",  # Keyboard and mouse control
    "Pillow>=9.0.0",  # Image processing
    "numpy>=1.22.0",  # Numerical processing
    "openai>=0.27.0",  # OpenAI API (optional)
    "anthropic>=0.2.0",  # Anthropic API (optional)
    "requests>=2.28.0",  # HTTP requests
    "aiohttp>=3.8.1",  # Async HTTP
    "python-dotenv>=0.21.0",  # Environment variable handling
    "psutil>=5.9.0",  # System monitoring
    "pytest>=7.0.0",  # Testing
]

# Add platform-specific dependencies
if sys.platform == "darwin":  # macOS
    dependencies.append("pyobjc-core>=8.5")
    dependencies.append("pyobjc-framework-Quartz>=8.5")
    dependencies.append("pyobjc-framework-Cocoa>=8.5")

setup(
    name="macagent",
    version="1.0.0",
    description="AI Agent for Mac Automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/MacAgent",
    packages=find_packages(),
    package_data={
        "MacAgent": [
            "config/*.json",
            "resources/**/*",
            "ui/assets/**/*",
        ],
    },
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'macagent=MacAgent.main:main',
            'macagent-cli=MacAgent.main:main_cli',
            'macagent-demo=MacAgent.main:main_demo',
        ],
    },
    include_package_data=True,
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Desktop Environment",
        "Topic :: Software Development :: User Interfaces",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    keywords="mac, automation, ai, agent, assistant, ui, interface",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/MacAgent/issues",
        "Source": "https://github.com/yourusername/MacAgent",
        "Documentation": "https://github.com/yourusername/MacAgent#readme",
    },
)

# Setup post-install script
if "install" in sys.argv:
    try:
        print("Running post-installation setup...")
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MacAgent", "config")
        os.makedirs(config_dir, exist_ok=True)
        
        # Create default config file if it doesn't exist
        api_keys_file = os.path.join(config_dir, "api_keys.json")
        if not os.path.exists(api_keys_file):
            with open(api_keys_file, "w") as f:
                f.write('{\n  "openai_api_key": "",\n  "anthropic_api_key": ""\n}')
        
        print("Setup complete! You can now run MacAgent using the 'macagent' command.")
    except Exception as e:
        print(f"Warning: Post-installation setup failed: {e}")
        print("You may need to create the config files manually.")

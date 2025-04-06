# MacAgent UI Components

This directory contains the user interface components for the MacAgent system, providing a clean and intuitive interface for interacting with the AI agent on macOS.

## Overview

The UI is built using PyQt5 and consists of three main components:

1. **CommandInterface**: Provides input methods for natural language instructions
2. **FeedbackSystem**: Displays real-time agent actions and understanding
3. **ConfigurationUI**: Allows customization of agent parameters and preferences

## Components

### CommandInterface

The CommandInterface provides a simple input method for sending instructions to the agent:

- Text and voice input modes
- Command history and suggestions
- System commands with `/` prefix
- Help and documentation features

### FeedbackSystem

The FeedbackSystem displays real-time feedback about agent actions:

- Activity log with different types of messages (info, actions, decisions, warnings, errors)
- Task progress tracking
- Visualization of the agent's screen understanding
- Configurable verbosity levels

### ConfigurationUI

The ConfigurationUI allows customization of the agent:

- Agent parameter adjustments
- User profiles for different use cases
- Permission management
- Theme customization
- System status monitoring

## Usage

### Running the Application

To run the complete application:

```bash
# From the project root
python -m MacAgent.src.ui.main_app
```

### Using Components Individually

You can also use the components individually in your own application:

```python
from MacAgent.src.ui.command_interface import CommandInterface
from MacAgent.src.ui.feedback_system import FeedbackSystem
from MacAgent.src.ui.configuration_ui import ConfigurationUI

# Create components
command_interface = CommandInterface()
feedback_system = FeedbackSystem()
config_ui = ConfigurationUI()

# Connect signals
command_interface.command_submitted.connect(lambda cmd: process_command(cmd))
```

## Customization

Each UI component can be customized and extended:

- Add custom command handlers to the CommandInterface
- Create custom feedback types in the FeedbackSystem
- Add new configuration options to the ConfigurationUI

## Requirements

- Python 3.8+
- PyQt5 5.15+
- Additional dependencies as specified in requirements.txt 
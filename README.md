# MacAgent - AI Agent for macOS

MacAgent is a modular framework for creating AI agents that can interact with macOS through screen capture and mouse/keyboard control. It provides a clean architecture for building autonomous agents that can perform tasks on your Mac.

## Features

- **Screen Perception**: Capture and analyze screen content
- **UI Interaction**: Control mouse and keyboard to interact with applications
- **Memory System**: Store and retrieve state information
- **Planning Module**: Generate and execute plans based on instructions
- **Modular Architecture**: Clean separation of concerns for easy extension
- **Interactive UI**: Chat interface to communicate with the agent
- **Advanced NLP**: Optional integration with OpenAI or Anthropic for better natural language understanding

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/MacAgent.git
   cd MacAgent
   ```

2. Install dependencies:
   ```
   # macOS users need PortAudio for PyAudio
   brew install portaudio
   
   # Install Python dependencies
   pip install -r MacAgent/requirements.txt
   ```

3. Set up the UI components:
   ```
   python MacAgent/setup_ui.py
   ```

4. (Optional) Configure API keys for advanced language understanding:
   ```
   # Edit the config/api_keys.json file with your API keys
   nano config/api_keys.json
   ```

## Advanced Natural Language Processing

MacAgent can optionally use OpenAI (GPT-4) or Anthropic (Claude) models to better understand natural language instructions. This significantly improves the agent's ability to handle complex requests.

To enable this feature:

1. Open the `config/api_keys.json` file
2. Add your OpenAI API key: `"openai_api_key": "your-api-key-here"`
3. Or add your Anthropic API key: `"anthropic_api_key": "your-api-key-here"`

With API keys configured, MacAgent will:
- Better parse complex instructions
- Break multi-step instructions into logical steps
- Handle ambiguous commands more intelligently

If no API keys are provided, MacAgent will fall back to basic rule-based parsing which works for simple commands.

## Project Structure

```
MacAgent/
├── src/
│   ├── core/              # Core functionality
│   │   ├── __init__.py    # Package initialization
│   │   ├── agent.py       # Main agent loop and coordination
│   │   ├── perception.py  # Screen capture and analysis
│   │   ├── action.py      # Mouse and keyboard control
│   │   ├── planning.py    # Instruction processing and planning
│   │   └── memory.py      # State storage and retrieval
│   ├── intelligence/      # Language understanding with LLMs
│   │   ├── llm_connector.py      # Connection to OpenAI/Anthropic
│   │   └── instruction_processor.py # Natural language processing
│   ├── ui/                # User interface components
│   │   ├── __init__.py    # UI package initialization
│   │   ├── main_app.py    # Main application window
│   │   └── command_interface.py # Command input interface
│   ├── utils/             # Utility functions
│   └── plugins/           # Optional plugins
├── config/                # Configuration files
│   └── api_keys.json      # API keys for OpenAI/Anthropic
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Usage

### Running with the Graphical User Interface (Recommended)

This launches the MacAgent with a chat interface where you can type instructions:

```bash
python -m MacAgent.main
```

### Running in Command Line Mode

You can also run MacAgent in command-line mode:

```bash
python -m MacAgent.main --cli
```

### Running the Demo Mode

To run with the original demo mode (processes a predefined instruction):

```bash
python -m MacAgent.main --demo
```

### Additional Options

Enable debug logging:

```bash
python -m MacAgent.main --debug
```

Disable file logging:

```bash
python -m MacAgent.main --no-file-log
```

## Development

The agent is built with a modular architecture:

1. **Perception Module**: Captures screenshots and analyzes UI elements
2. **Action Module**: Controls mouse and keyboard actions
3. **Planning Module**: Processes instructions and generates plans
4. **Memory System**: Stores observations and action history
5. **Agent Loop**: Coordinates all the components
6. **UI Components**: Provides user interface for interaction

To extend the agent, you can:
- Add new action handlers
- Improve perception capabilities
- Enhance planning strategies
- Create plugins for specific applications
- Customize the user interface

## Requirements

- Python 3.8+
- macOS (tested on Ventura and later)
- PyQt5 (for the user interface)
- Various Python packages (see requirements.txt)

## Troubleshooting

If you encounter a "division by zero" error in the logs, this has been fixed in the latest version. Make sure you're running the latest code.

If the UI doesn't appear when running the application, make sure you have PyQt5 installed:

```bash
pip install PyQt5>=5.15.9
```

Or run the setup script:

```bash
python MacAgent/setup_ui.py
```

## License

MIT License

## Disclaimer

This tool is for educational and research purposes only. Use responsibly and in accordance with Apple's terms of service and relevant laws and regulations.

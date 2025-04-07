# MacAgent - AI Agent for macOS

MacAgent is a modular framework for creating AI agents that can interact with macOS through screen capture and mouse/keyboard control. It provides a clean architecture for building autonomous agents that can perform tasks on your Mac using natural language instructions.

## Features

- **Screen Perception**: Capture and analyze screen content
- **Robust UI Interaction**: Control mouse and keyboard with reliable parameter validation
- **Intelligent Planning**: Break down complex tasks into manageable steps
- **Memory System**: Store and retrieve state information
- **Modular Architecture**: Clean separation of concerns for easy extension
- **Interactive UI**: Chat interface to communicate with the agent
- **Advanced NLP**: Integration with OpenAI or Anthropic for better natural language understanding
- **Comprehensive Error Handling**: Graceful recovery from errors during execution
- **Thorough Parameter Validation**: Smart filtering of parameters to ensure compatibility

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

MacAgent uses advanced language models (OpenAI's GPT or Anthropic's Claude) to understand complex natural language instructions and extract relevant parameters for actions.

To enable this feature:

1. Open the `config/api_keys.json` file
2. Add your OpenAI API key: `"openai_api_key": "your-api-key-here"`
3. Or add your Anthropic API key: `"anthropic_api_key": "your-api-key-here"`

With API keys configured, MacAgent will:
- Parse complex instructions with high accuracy
- Break multi-step instructions into logical steps
- Handle ambiguous commands more intelligently
- Extract and validate parameters for actions
- Filter out invalid parameters to ensure reliable execution

If no API keys are provided, MacAgent will fall back to basic rule-based parsing which works for simple commands.

## Project Structure

```
MacAgent/
├── src/
│   ├── core/              # Core functionality
│   │   ├── __init__.py    # Package initialization
│   │   ├── agent.py       # Main agent loop and coordination
│   │   ├── perception.py  # Screen capture and analysis
│   │   ├── action.py      # Mouse and keyboard control with parameter filtering
│   │   ├── planning.py    # Instruction processing and planning
│   │   ├── task_manager.py # Task tracking and monitoring
│   │   └── memory.py      # State storage and retrieval
│   ├── intelligence/      # Language understanding with LLMs
│   │   ├── llm_connector.py      # Connection to OpenAI/Anthropic
│   │   └── instruction_processor.py # Natural language processing with parameter validation
│   ├── ui/                # User interface components
│   │   ├── __init__.py    # UI package initialization
│   │   ├── main_app.py    # Main application window
│   │   └── command_interface.py # Command input interface
│   ├── utils/             # Utility functions
│   └── plugins/           # Optional plugins
├── config/                # Configuration files
│   └── api_keys.json      # API keys for OpenAI/Anthropic
├── tests/                 # Test suite
│   └── test_integration.py # Integration tests for components
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

You can run MacAgent in command-line mode for a text-based interface:

```bash
python -m MacAgent.main --cli
```

Example commands you can use:
- "Take a screenshot of the desktop"
- "Open Safari and go to google.com"
- "Click on the Apple menu in the top left"
- "Type 'Hello world' in the current application"
- "Move the mouse to position x=500, y=300"

### Running the Demo Mode

To run with the demo mode (processes a predefined instruction):

```bash
python -m MacAgent.main --demo
```

### Loading a Configuration File

You can specify a custom configuration file:

```bash
python -m MacAgent.main --config path/to/config.json
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

## Development and Extension

MacAgent is designed with a modular architecture that makes it easy to extend:

### Adding New Actions

1. Open `src/core/action.py`
2. Add a new method to the `ActionModule` class
3. Ensure it accepts appropriate parameters and uses the parameter filtering system

Example of a custom action:
```python
async def my_custom_action(self, param1: str, param2: int = 0, **kwargs) -> bool:
    """
    Custom action that does something cool.
    
    Args:
        param1: First parameter
        param2: Second parameter with default value
        
    Returns:
        True if successful, False otherwise
    """
    # The **kwargs parameter ensures compatibility with the parameter filtering system
    
    # Your implementation here
    return True
```

### Enhancing the Instruction Processor

1. Open `src/intelligence/instruction_processor.py`
2. Modify the parameter schema in `_get_parameter_schema`
3. Update the validation logic in `validate_action_parameters`

### Implementing New UI Features

1. Locate the UI components in `src/ui/`
2. Modify the existing components or add new ones
3. Update the main application to include your new features

## Running Tests

MacAgent includes a comprehensive test suite to verify that all components work correctly:

```bash
python -m unittest discover -s MacAgent/tests
```

To run a specific test file:

```bash
python -m MacAgent.tests.test_integration
```

## Requirements

- Python 3.8+
- macOS (tested on Ventura and later)
- PyQt5 (for the user interface)
- pynput, pyautogui (for mouse and keyboard control)
- asyncio (for asynchronous operations)
- OpenAI or Anthropic API key (optional, for advanced language understanding)

## Troubleshooting

### Common Issues

1. **Parameter Validation Errors**
   
   If you see warnings about unexpected parameters, this is normal. The parameter filtering system automatically removes invalid parameters. If you need to add support for a new parameter, update the `validate_action_parameters` method in the instruction processor.

2. **Task Execution Errors**
   
   If tasks fail to execute, check the logs for details. The new error handling system provides clear information about what went wrong.

3. **UI Issues**
   
   If the UI doesn't appear when running the application, make sure you have PyQt5 installed:

   ```bash
   pip install PyQt5>=5.15.9
   ```

   Or run the setup script:

   ```bash
   python MacAgent/setup_ui.py
   ```

4. **Language Model Connection Issues**
   
   If you experience problems with language model connections, verify your API keys and internet connection. The application will fall back to basic parsing if API connections fail.

5. **Asyncio Errors**
   
   If you see asyncio-related errors, make sure you're not mixing async and sync code without proper handling. The application uses asyncio for all operations.

## License

MIT License

## Disclaimer

This tool is for educational and research purposes only. Use responsibly and in accordance with Apple's terms of service and relevant laws and regulations.

"""
CommandInterface Module

This module provides the command input interface for MacAgent.
"""

import os
import json
from enum import Enum, auto
from typing import List, Callable, Dict, Optional, Any
import logging

try:
    import speech_recognition as sr
except ImportError:
    sr = None

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                            QPushButton, QTextEdit, QComboBox, QLabel,
                            QCompleter, QScrollArea, QFrame, QToolTip)
from PyQt5.QtCore import Qt, QStringListModel, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QFontMetrics, QKeySequence
from PyQt5.QtGui import QColor, QPalette

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """Enum representing different input modes for commands."""
    TEXT = auto()
    VOICE = auto()


class CommandInterface(QWidget):
    """
    A user interface for entering and managing commands for the AI agent.
    
    Features:
    - Text and voice input modes
    - Command history tracking
    - Command suggestions
    - Real-time feedback
    - Help documentation
    """
    
    # Signal emitted when a command is submitted
    command_submitted = pyqtSignal(str)
    
    # Signal emitted when input mode changes
    mode_changed = pyqtSignal(InputMode)
    
    def __init__(self, parent=None):
        """Initialize the command interface."""
        super().__init__(parent)
        
        self.command_history: List[str] = []
        self.history_index: int = -1
        self.max_history: int = 100
        self.suggestions: List[str] = []
        self.input_mode: InputMode = InputMode.TEXT
        self.command_handlers: Dict[str, Callable] = {}
        self.help_docs: Dict[str, str] = {}
        
        self._load_history()
        self._load_suggestions()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Set up fonts
        title_font = QFont("Helvetica", 16, QFont.Bold)
        regular_font = QFont("Helvetica", 12)
        
        # Title
        title_label = QLabel("MacAgent Command Interface")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # History display
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        self.history_display.setFont(regular_font)
        self.history_display.setMinimumHeight(200)
        self.history_display.setPlaceholderText("Command history will appear here")
        main_layout.addWidget(self.history_display)
        
        # Input mode selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Input Mode:")
        mode_label.setFont(regular_font)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([mode.name.capitalize() for mode in InputMode])
        self.mode_selector.setCurrentIndex(0)
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_selector)
        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)
        
        # Command input
        input_layout = QHBoxLayout()
        
        self.command_input = QLineEdit()
        self.command_input.setFont(regular_font)
        self.command_input.setPlaceholderText("Enter a command or query...")
        self.command_input.returnPressed.connect(self._on_submit)
        
        # Set up autocompleter
        self.completer = QCompleter(self.suggestions)
        self.completer_model = QStringListModel(self.suggestions)
        self.completer.setModel(self.completer_model)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.command_input.setCompleter(self.completer)
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self._on_submit)
        
        self.voice_button = QPushButton()
        self.voice_button.setIcon(QIcon.fromTheme("audio-input-microphone"))
        self.voice_button.setToolTip("Click and hold to record voice input")
        self.voice_button.setFixedSize(QSize(40, 40))
        self.voice_button.pressed.connect(self._start_voice_recognition)
        self.voice_button.released.connect(self._stop_voice_recognition)
        
        self.help_button = QPushButton("?")
        self.help_button.setFixedSize(QSize(30, 30))
        self.help_button.setToolTip("Show help and documentation")
        self.help_button.clicked.connect(self._show_help)
        
        input_layout.addWidget(self.command_input)
        input_layout.addWidget(self.voice_button)
        input_layout.addWidget(self.submit_button)
        input_layout.addWidget(self.help_button)
        
        main_layout.addLayout(input_layout)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignRight)
        self.status_label.setFont(QFont("Helvetica", 10, QFont.StyleItalic))
        main_layout.addWidget(self.status_label)
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Focus the input field
        self.command_input.setFocus()
        
    def _on_submit(self):
        """Handle command submission."""
        command = self.command_input.text().strip()
        if not command:
            return
            
        # Add to history and clear input
        self._add_to_history(command)
        self.command_input.clear()
        
        # Display in history view
        self.history_display.append(f"> {command}")
        
        # Process the command
        self._process_command(command)
        
        # Emit signal for external handlers
        self.command_submitted.emit(command)
        
    def _process_command(self, command: str):
        """Process the submitted command."""
        # Command handling logic
        if command.startswith("/"):
            # System commands
            parts = command[1:].split()
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if cmd in self.command_handlers:
                try:
                    self.command_handlers[cmd](args)
                except Exception as e:
                    self._show_error(f"Error executing command: {e}")
            else:
                self._show_error(f"Unknown command: /{cmd}")
        else:
            # Regular instruction to the agent
            self.status_label.setText("Processing...")
            # Processing happens via the signal which other components listen to
    
    def _add_to_history(self, command: str):
        """Add a command to the history."""
        # Don't add duplicates consecutively
        if self.command_history and self.command_history[-1] == command:
            return
            
        self.command_history.append(command)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
            
        self.history_index = len(self.command_history)
        self._save_history()
        
        # Add to suggestions if appropriate
        if command not in self.suggestions:
            self.suggestions.append(command)
            self.completer_model.setStringList(self.suggestions)
            self._save_suggestions()
    
    def _on_mode_changed(self, index: int):
        """Handle input mode changes."""
        mode_name = self.mode_selector.currentText().upper()
        self.input_mode = InputMode[mode_name]
        
        # Update UI based on mode
        self.voice_button.setEnabled(self.input_mode == InputMode.VOICE)
        
        # Emit signal for external handlers
        self.mode_changed.emit(self.input_mode)
        
        # Update status
        self.status_label.setText(f"Mode: {mode_name.capitalize()}")
    
    def _start_voice_recognition(self):
        """Start voice recognition."""
        if not sr:
            self._show_error("Speech recognition is not available. Please install the required package.")
            return
            
        self.status_label.setText("Listening...")
        self.command_input.setPlaceholderText("Listening...")
        
        # Voice recognition happens in a separate thread
        # For demonstration, we're just showing the UI feedback
    
    def _stop_voice_recognition(self):
        """Stop voice recognition and process the result."""
        if not sr:
            return
            
        self.status_label.setText("Processing speech...")
        self.command_input.setPlaceholderText("Processing speech...")
        
        # Here we would normally get the result from the speech recognition
        # For demonstration, we're just simulating a delay and a result
        self.command_input.setText("This is a simulated voice command")
        self.status_label.setText("Ready")
        self.command_input.setPlaceholderText("Enter a command or query...")
        
    def _show_help(self):
        """Show help and documentation."""
        help_text = """
        <h3>MacAgent Command Help</h3>
        
        <p><b>Basic Usage:</b><br>
        Type your instructions in natural language and press Enter or click Submit.</p>
        
        <p><b>System Commands:</b><br>
        /help - Show this help text<br>
        /clear - Clear command history<br>
        /config - Open configuration panel<br>
        /history - Show full command history</p>
        
        <p><b>Voice Input:</b><br>
        Select Voice mode and press the microphone button to start recording.
        Release to process the voice input.</p>
        
        <p><b>Examples:</b><br>
        "Take a screenshot of this window"<br>
        "Open Safari and go to apple.com"<br>
        "Find all PDF files in my Downloads folder"</p>
        """
        
        # Create a scrollable dialog for help
        help_dialog = QScrollArea(self)
        help_dialog.setWindowTitle("MacAgent Help")
        help_dialog.setMinimumSize(500, 400)
        
        help_content = QLabel(help_text)
        help_content.setWordWrap(True)
        help_content.setTextFormat(Qt.RichText)
        help_content.setMargin(10)
        
        help_dialog.setWidget(help_content)
        help_dialog.setWidgetResizable(True)
        help_dialog.show()
    
    def _show_error(self, message: str):
        """Show an error message in the interface."""
        self.status_label.setText(f"Error: {message}")
        self.history_display.append(f"<span style='color:red'>Error: {message}</span>")
    
    def _load_history(self):
        """Load command history from file."""
        history_file = os.path.expanduser("~/.macagent/command_history.json")
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.command_history = json.load(f)
                self.history_index = len(self.command_history)
        except Exception as e:
            logger.error(f"Failed to load command history: {e}")
    
    def _save_history(self):
        """Save command history to file."""
        history_dir = os.path.expanduser("~/.macagent")
        history_file = os.path.join(history_dir, "command_history.json")
        try:
            os.makedirs(history_dir, exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump(self.command_history, f)
        except Exception as e:
            logger.error(f"Failed to save command history: {e}")
    
    def _load_suggestions(self):
        """Load command suggestions from file."""
        suggestions_file = os.path.expanduser("~/.macagent/command_suggestions.json")
        try:
            if os.path.exists(suggestions_file):
                with open(suggestions_file, 'r') as f:
                    self.suggestions = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load command suggestions: {e}")
    
    def _save_suggestions(self):
        """Save command suggestions to file."""
        suggestions_dir = os.path.expanduser("~/.macagent")
        suggestions_file = os.path.join(suggestions_dir, "command_suggestions.json")
        try:
            os.makedirs(suggestions_dir, exist_ok=True)
            with open(suggestions_file, 'w') as f:
                json.dump(self.suggestions, f)
        except Exception as e:
            logger.error(f"Failed to save command suggestions: {e}")
    
    def register_command_handler(self, command: str, handler: Callable):
        """Register a handler function for a specific command."""
        self.command_handlers[command] = handler
    
    def add_help_documentation(self, command: str, doc: str):
        """Add help documentation for a command."""
        self.help_docs[command] = doc
    
    def clear_history_display(self):
        """Clear the history display."""
        self.history_display.clear()
    
    def set_status(self, message: str):
        """Set the status message."""
        self.status_label.setText(message)
    
    def add_feedback(self, message: str):
        """Add feedback to the history display."""
        self.history_display.append(message)

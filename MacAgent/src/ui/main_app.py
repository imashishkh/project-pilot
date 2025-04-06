"""
Main Application for MacAgent UI

This module provides a complete UI application for the MacAgent.
"""

import sys
import os
import logging
from typing import Dict, Any
import asyncio
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QSplitter, QTabWidget, QAction, QMenu,
                           QToolBar, QStatusBar, QLabel, QMessageBox, QPushButton,
                           QDialog, QTextEdit)
from PyQt5.QtCore import Qt, QSettings, QSize, QTimer
from PyQt5.QtGui import QIcon, QFont, QKeySequence

from .command_interface import CommandInterface, InputMode
from .feedback_system import FeedbackSystem, VerbosityLevel
from .configuration_ui import ConfigurationUI

# Import from core module for agent integration
from MacAgent.src.core import AgentLoop, AgentConfig
from MacAgent.src.core.planning import PlanStatus

# Import permissions checker
try:
    from MacAgent.src.utils.permissions_checker import permission_checker
except ImportError:
    permission_checker = None

logger = logging.getLogger(__name__)


class PermissionsDialog(QDialog):
    """Dialog for displaying and managing macOS permissions."""
    
    def __init__(self, permissions_status, parent=None):
        """Initialize the permissions dialog."""
        super().__init__(parent)
        self.permissions_status = permissions_status
        self.setWindowTitle("MacAgent Permissions")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout()
        
        # Title and description
        title = QLabel("MacAgent Permissions")
        title.setFont(QFont("Helvetica", 16, QFont.Bold))
        layout.addWidget(title)
        
        description = QLabel(
            "MacAgent requires certain permissions to control your Mac. "
            "Please review and grant the following permissions:"
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Permissions guide text
        if permission_checker:
            guide_text = permission_checker.display_permissions_guide()
        else:
            guide_text = (
                "MacAgent requires the following permissions:\n\n"
                "- Accessibility: For controlling mouse and keyboard\n"
                "- Screen Recording: For capturing screen contents\n"
                "- Automation: For controlling other applications\n\n"
                "Please grant these permissions in System Preferences > Security & Privacy > Privacy."
            )
        
        guide = QTextEdit()
        guide.setReadOnly(True)
        guide.setPlainText(guide_text)
        layout.addWidget(guide)
        
        # Open Preferences buttons
        buttons_layout = QHBoxLayout()
        
        if permission_checker:
            # Only show these buttons if the permission checker is available
            accessibility_btn = QPushButton("Open Accessibility Settings")
            accessibility_btn.clicked.connect(permission_checker.open_accessibility_preferences)
            buttons_layout.addWidget(accessibility_btn)
            
            screen_recording_btn = QPushButton("Open Screen Recording Settings")
            screen_recording_btn.clicked.connect(permission_checker.open_screen_recording_preferences)
            buttons_layout.addWidget(screen_recording_btn)
            
            automation_btn = QPushButton("Open Automation Settings")
            automation_btn.clicked.connect(permission_checker.open_automation_preferences)
            buttons_layout.addWidget(automation_btn)
        
        layout.addLayout(buttons_layout)
        
        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)
        
        self.setLayout(layout)


class MacAgentApp(QMainWindow):
    """
    Main application window for MacAgent.
    
    This class integrates all UI components into a complete application.
    """
    
    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        
        # Setup logging
        self._setup_logging()
        
        # Check permissions
        self._check_permissions()
        
        # Initialize agent core
        self.agent_config = AgentConfig()
        self.agent = AgentLoop(self.agent_config)
        
        # Initialize UI components
        self.command_interface = CommandInterface()
        self.feedback_system = FeedbackSystem()
        self.config_ui = ConfigurationUI()
        
        # Connect signals
        self._connect_signals()
        
        # Setup UI
        self._setup_ui()
        
        # Apply initial configuration
        self._apply_config(self.config_ui.get_current_config())
        
        # Show window
        self.show()
        
    def _setup_logging(self):
        """Setup application logging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.expanduser('~/.macagent/app.log'))
            ]
        )
    
    def _check_permissions(self):
        """Check and request necessary macOS permissions."""
        if permission_checker:
            permissions_status = permission_checker.check_all_permissions()
            
            # If any permission is not granted, show the permissions dialog
            if not all(permissions_status.values()):
                dialog = PermissionsDialog(permissions_status, self)
                dialog.exec_()
                
                # After dialog is closed, check again
                updated_status = permission_checker.check_all_permissions()
                if not all(updated_status.values()):
                    logger.warning("Some permissions are still not granted. Functionality may be limited.")
        else:
            logger.warning("Permission checker not available. Permissions not verified.")
    
    def _setup_ui(self):
        """Setup the main UI."""
        # Set window properties
        self.setWindowTitle("MacAgent")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for main components
        splitter = QSplitter(Qt.Vertical)
        
        # Add command interface to the top
        splitter.addWidget(self.command_interface)
        
        # Add feedback system to the bottom
        splitter.addWidget(self.feedback_system)
        
        # Set initial sizes (command interface smaller than feedback system)
        splitter.setSizes([200, 400])
        
        main_layout.addWidget(splitter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create status bar
        self._create_status_bar()
        
        # Create toolbar
        self._create_tool_bar()
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # Settings action
        settings_action = QAction("&Settings", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        # Toggle command interface
        toggle_cmd_action = QAction("&Command Interface", self)
        toggle_cmd_action.setCheckable(True)
        toggle_cmd_action.setChecked(True)
        toggle_cmd_action.triggered.connect(
            lambda checked: self.command_interface.setVisible(checked))
        view_menu.addAction(toggle_cmd_action)
        
        # Toggle feedback system
        toggle_feedback_action = QAction("&Feedback System", self)
        toggle_feedback_action.setCheckable(True)
        toggle_feedback_action.setChecked(True)
        toggle_feedback_action.triggered.connect(
            lambda checked: self.feedback_system.setVisible(checked))
        view_menu.addAction(toggle_feedback_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_status_bar(self):
        """Create the status bar."""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Add status labels
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label, 1)
        
        # Add model info
        self.model_label = QLabel("Model: GPT-4")
        status_bar.addPermanentWidget(self.model_label)
    
    def _create_tool_bar(self):
        """Create the toolbar."""
        tool_bar = QToolBar("Main Toolbar")
        tool_bar.setMovable(False)
        self.addToolBar(tool_bar)
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._show_settings)
        tool_bar.addAction(settings_action)
        
        tool_bar.addSeparator()
        
        # Clear action
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self._clear_interface)
        tool_bar.addAction(clear_action)
    
    def _connect_signals(self):
        """Connect signals between components."""
        # Command interface signals
        self.command_interface.command_submitted.connect(self._on_command_submitted)
        self.command_interface.mode_changed.connect(self._on_mode_changed)
        
        # Feedback system signals
        self.feedback_system.preferences_changed.connect(self._on_feedback_preferences_changed)
        
        # Configuration UI signals
        self.config_ui.config_changed.connect(self._apply_config)
        
        # Register command handlers
        self.command_interface.register_command_handler("help", self._handle_help_command)
        self.command_interface.register_command_handler("clear", self._handle_clear_command)
        self.command_interface.register_command_handler("config", self._handle_config_command)
    
    def _on_command_submitted(self, command: str):
        """Handle command submission."""
        # Log the command
        logger.info(f"Command submitted: {command}")
        
        # Update status
        self.status_label.setText("Processing command...")
        
        # Display in feedback system
        self.feedback_system.add_action(f"Processing command: {command}")
        
        # Create an async task to process the command with the agent core
        self.processing_task = asyncio.create_task(self._process_command_with_agent(command))
    
    async def _process_command_with_agent(self, command: str):
        """Process a command using the agent core."""
        try:
            # Create a plan from the command
            plan = await self.agent.process_instruction(command)
            
            if plan:
                # Show plan diagnostic information
                self.feedback_system.add_diagnostic("Generated Plan", {
                    "Instruction": plan.instruction,
                    "Total Steps": len(plan.steps),
                    "Plan ID": plan.plan_id,
                    "Steps": [f"{i+1}. {step.description} ({step.action_type})" 
                             for i, step in enumerate(plan.steps)]
                })
                
                # Execute the plan
                self.agent.start()
                
                # Update the UI when finished
                # This is a simplified approach; ideally we'd use signals/callbacks
                await asyncio.sleep(1)  # Give the agent some time to start processing
                
                # Check status periodically
                for _ in range(30):  # Check for up to 30 seconds
                    await asyncio.sleep(1)
                    
                    # If the plan is completed or failed, show status
                    if not self.agent.current_plan or self.agent.current_plan.status in [
                        PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED
                    ]:
                        if not self.agent.current_plan:
                            status = "completed"
                            self.feedback_system.add_info(f"Plan execution {status}")
                        elif self.agent.current_plan.status == PlanStatus.COMPLETED:
                            status = "completed"
                            self.feedback_system.add_info(f"Plan execution {status}")
                        else:
                            status = "failed"
                            # If failed, show more detailed diagnostic information
                            failed_steps = [step for step in plan.steps if step.status == PlanStatus.FAILED]
                            
                            if failed_steps:
                                self.feedback_system.add_error(f"Plan execution {status}", {
                                    "Failed steps": [f"{step.description}: {step.error}" for step in failed_steps]
                                })
                                
                                # Add detailed diagnostics
                                self.feedback_system.add_diagnostic("Plan Execution Failure", {
                                    "Status": plan.status.name,
                                    "Failed Steps": [
                                        f"{step.description} ({step.action_type}): {step.error}" 
                                        for step in failed_steps
                                    ],
                                    "Execution Time": f"{time.time() - (plan.start_time or 0):.2f} seconds",
                                    "Permissions": self._check_agent_permissions() if permission_checker else "Unknown"
                                })
                            else:
                                self.feedback_system.add_error(f"Plan execution {status}", {
                                    "reason": "Unknown failure"
                                })
                        
                        self.status_label.setText("Ready")
                        break
                else:
                    # If the loop completed without breaking, it timed out
                    self.feedback_system.add_info("Plan execution taking longer than expected")
            else:
                self.feedback_system.add_error("Failed to create plan from instruction", 
                                             {"reason": "Unable to parse instruction"})
                self.status_label.setText("Ready")
        
        except Exception as e:
            # Log the error
            logger.error(f"Error processing command with agent: {str(e)}")
            self.feedback_system.add_error("Error processing command", 
                                         {"reason": str(e)})
            
            # Add detailed diagnostic information
            import traceback
            self.feedback_system.add_diagnostic("Command Processing Error", {
                "Error": str(e),
                "Traceback": traceback.format_exc().split('\n'),
                "Command": command,
                "Permissions": self._check_agent_permissions() if permission_checker else "Unknown"
            })
            
            self.status_label.setText("Error")
        
        # Set focus back to command input
        self.command_interface.command_input.setFocus()
    
    def _on_mode_changed(self, mode: InputMode):
        """Handle input mode changes."""
        logger.info(f"Input mode changed to {mode.name}")
        self.status_label.setText(f"Input mode: {mode.name}")
    
    def _on_feedback_preferences_changed(self, preferences: Dict[str, Any]):
        """Handle feedback preferences changes."""
        logger.info(f"Feedback preferences changed: {preferences}")
        
        # Apply verbosity changes
        if "verbosity_level" in preferences:
            verbosity = preferences["verbosity_level"]
            logger.info(f"Setting verbosity to {verbosity}")
    
    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration settings to the application."""
        logger.info("Applying configuration settings")
        
        # Update model label
        self.model_label.setText(f"Model: {config.get('model', 'GPT-4')}")
        
        # Apply verbosity setting
        verbosity = config.get("verbosity", 2)
        verbosity_levels = [
            VerbosityLevel.MINIMAL,
            VerbosityLevel.NORMAL,
            VerbosityLevel.DETAILED,
            VerbosityLevel.DEBUG
        ]
        if 1 <= verbosity <= 4:
            self.feedback_system.verbosity_level = verbosity_levels[verbosity - 1]
        
        # Apply theme (simplified)
        appearance = config.get("appearance", {})
        theme = appearance.get("theme", "System Default")
        logger.info(f"Setting theme to {theme}")
        
        # In a real app, this would apply the theme to the application style
    
    def _show_settings(self):
        """Show the settings dialog."""
        self.config_ui.show()
    
    def _show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About MacAgent",
            """<h3>MacAgent</h3>
            <p>Version 1.0</p>
            <p>An AI assistant for Mac that helps automate tasks through natural language.</p>
            <p>&copy; 2024 MacAgent Team</p>"""
        )
    
    def _clear_interface(self):
        """Clear the interface."""
        self.command_interface.clear_history_display()
        self.feedback_system.clear_log()
        self.status_label.setText("Ready")
    
    def _handle_help_command(self, args):
        """Handle the /help command."""
        self.command_interface._show_help()
    
    def _handle_clear_command(self, args):
        """Handle the /clear command."""
        self._clear_interface()
    
    def _handle_config_command(self, args):
        """Handle the /config command."""
        self._show_settings()

    def _simulate_response(self, command: str):
        """
        Legacy method for simulating responses before agent integration.
        This is kept for backward compatibility with demo mode.
        """
        # In actual usage, we now use _process_command_with_agent instead
        if "screenshot" in command.lower():
            self.feedback_system.add_action("Taking screenshot")
            self.feedback_system.add_info("Screenshot captured successfully")
            
            # Simulate task tracking
            task_id = "screenshot_task"
            self.feedback_system.start_task(task_id, "Process screenshot", 
                                          ["Capture screen", "Analyze content", "Identify UI elements"])
            
            # Simulate progress
            QTimer.singleShot(500, lambda: self.feedback_system.advance_task_step(task_id))
            QTimer.singleShot(1000, lambda: self.feedback_system.advance_task_step(task_id))
            QTimer.singleShot(1500, lambda: self.feedback_system.complete_task(task_id))
            
        elif "find" in command.lower():
            self.feedback_system.add_action("Searching files")
            self.feedback_system.add_info("Search completed")
            
            # Simulate decision making
            self.feedback_system.add_decision(
                "Using Spotlight to search for files",
                "Spotlight is faster for file system searches than manual directory traversal",
                ["Use find command", "Traverse directories manually", "Use Spotlight"]
            )
            
        elif "error" in command.lower():
            # Demonstrate error handling
            self.feedback_system.add_error("Failed to process command", 
                                         {"reason": "Simulated error for demonstration"})
        else:
            # Generic response
            self.feedback_system.add_info("Command processed successfully")
        
        # Update status
        self.status_label.setText("Ready")
        
        # Set focus back to command input
        self.command_interface.command_input.setFocus()

    def _check_agent_permissions(self) -> Dict[str, bool]:
        """
        Check the current status of permissions needed by the agent.
        
        Returns:
            Dictionary of permission statuses
        """
        if not permission_checker:
            return {"status": "Permission checker not available"}
        
        return permission_checker.check_all_permissions()


def main():
    """Main entry point for the MacAgent UI application."""
    app = QApplication(sys.argv)
    
    # Set application-wide font
    app.setFont(QFont("Helvetica", 12))
    
    # Create and show main window
    main_window = MacAgentApp()
    main_window.setGeometry(100, 100, 1000, 800)
    main_window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
"""
FeedbackSystem Module

This module provides real-time feedback about agent actions and decisions.
"""

import os
import logging
import time
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QTextEdit, QProgressBar, QComboBox, QCheckBox,
                           QGroupBox, QScrollArea, QSplitter, QPushButton,
                           QTabWidget, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QImage

logger = logging.getLogger(__name__)


class VerbosityLevel(Enum):
    """Enum representing different verbosity levels for feedback."""
    MINIMAL = auto()  # Essential information only
    NORMAL = auto()   # Regular amount of information
    DETAILED = auto() # Detailed information including decision logic
    DEBUG = auto()    # Maximum verbosity with technical details


class FeedbackSystem(QWidget):
    """
    A system for providing real-time feedback about agent actions and decisions.
    
    Features:
    - Display of agent actions and decisions
    - Screen visualization
    - Progress indicators
    - Different verbosity levels
    - Error and warning notifications
    """
    
    # Signal when feedback preferences change
    preferences_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the feedback system."""
        super().__init__(parent)
        
        self.verbosity_level = VerbosityLevel.NORMAL
        self.max_log_entries = 1000
        self.show_timestamps = True
        self.auto_scroll = True
        self.tasks: Dict[str, Dict[str, Any]] = {}  # Track ongoing tasks
        
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
        small_font = QFont("Helvetica", 10)
        
        # Title
        title_label = QLabel("MacAgent Feedback")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tab widget for different feedback views
        self.tab_widget = QTabWidget()
        
        # Main feedback log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        # Feedback log
        self.feedback_log = QTextEdit()
        self.feedback_log.setReadOnly(True)
        self.feedback_log.setFont(regular_font)
        self.feedback_log.setMinimumHeight(200)
        log_layout.addWidget(self.feedback_log)
        
        # Current activity display
        activity_group = QGroupBox("Current Activity")
        activity_group.setFont(small_font)
        activity_layout = QVBoxLayout(activity_group)
        
        self.activity_label = QLabel("No active tasks")
        self.activity_label.setFont(regular_font)
        activity_layout.addWidget(self.activity_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        activity_layout.addWidget(self.progress_bar)
        
        log_layout.addWidget(activity_group)
        
        # Add log tab
        self.tab_widget.addTab(log_tab, "Activity Log")
        
        # Screen visualization tab
        screen_tab = QWidget()
        screen_layout = QVBoxLayout(screen_tab)
        
        self.screen_view = QLabel("Screen visualization will appear here")
        self.screen_view.setAlignment(Qt.AlignCenter)
        self.screen_view.setMinimumHeight(300)
        self.screen_view.setFrameShape(QFrame.StyledPanel)
        screen_layout.addWidget(self.screen_view)
        
        # Screen view controls
        screen_controls = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh View")
        self.refresh_button.clicked.connect(self._refresh_screen_view)
        
        self.highlight_objects_checkbox = QCheckBox("Highlight Objects")
        self.highlight_objects_checkbox.setChecked(True)
        self.highlight_objects_checkbox.toggled.connect(self._on_highlight_toggled)
        
        screen_controls.addWidget(self.refresh_button)
        screen_controls.addWidget(self.highlight_objects_checkbox)
        screen_controls.addStretch()
        
        screen_layout.addLayout(screen_controls)
        
        # Add screen tab
        self.tab_widget.addTab(screen_tab, "Screen Visualization")
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Verbosity setting
        verbosity_group = QGroupBox("Verbosity Level")
        verbosity_layout = QVBoxLayout(verbosity_group)
        
        self.verbosity_selector = QComboBox()
        self.verbosity_selector.addItems([level.name.capitalize() for level in VerbosityLevel])
        self.verbosity_selector.setCurrentIndex(1)  # Default to NORMAL
        self.verbosity_selector.currentIndexChanged.connect(self._on_verbosity_changed)
        
        verbosity_layout.addWidget(self.verbosity_selector)
        settings_layout.addWidget(verbosity_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.timestamp_checkbox = QCheckBox("Show Timestamps")
        self.timestamp_checkbox.setChecked(self.show_timestamps)
        self.timestamp_checkbox.toggled.connect(self._on_timestamp_toggled)
        
        self.autoscroll_checkbox = QCheckBox("Auto-scroll Log")
        self.autoscroll_checkbox.setChecked(self.auto_scroll)
        self.autoscroll_checkbox.toggled.connect(self._on_autoscroll_toggled)
        
        display_layout.addWidget(self.timestamp_checkbox)
        display_layout.addWidget(self.autoscroll_checkbox)
        
        settings_layout.addWidget(display_group)
        settings_layout.addStretch()
        
        # Add settings tab
        self.tab_widget.addTab(settings_tab, "Settings")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Welcome message
        self.add_info("Feedback system initialized. Ready to provide insights about agent actions.")
        
    def add_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add an informational message to the feedback log."""
        self._add_log_entry("INFO", message, context)
    
    def add_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Add an action to the feedback log."""
        self._add_log_entry("ACTION", action, details)
    
    def add_decision(self, decision: str, reasoning: Optional[str] = None, 
                   alternatives: Optional[List[str]] = None):
        """Add a decision with reasoning to the feedback log."""
        context = {}
        if reasoning and self.verbosity_level in [VerbosityLevel.DETAILED, VerbosityLevel.DEBUG]:
            context["reasoning"] = reasoning
        if alternatives and self.verbosity_level == VerbosityLevel.DEBUG:
            context["alternatives"] = alternatives
            
        self._add_log_entry("DECISION", decision, context)
    
    def add_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add a warning to the feedback log."""
        self._add_log_entry("WARNING", message, context, color="orange")
    
    def add_error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add an error to the feedback log."""
        self._add_log_entry("ERROR", message, context, color="red")
    
    def _add_log_entry(self, entry_type: str, message: str, 
                     context: Optional[Dict[str, Any]] = None, color: str = None):
        """Add a log entry to the feedback log."""
        import datetime
        
        # Prepare entry
        timestamp = ""
        if self.show_timestamps:
            timestamp = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            
        type_str = f"[{entry_type}] "
        
        # Color formatting
        if color:
            log_entry = f"{timestamp}<span style='color:{color};'>{type_str}{message}</span>"
        else:
            log_entry = f"{timestamp}{type_str}{message}"
            
        # Add context for detailed verbosity
        if context and self.verbosity_level in [VerbosityLevel.DETAILED, VerbosityLevel.DEBUG]:
            context_str = "<ul>"
            for key, value in context.items():
                context_str += f"<li><b>{key}</b>: {value}</li>"
            context_str += "</ul>"
            log_entry += context_str
            
        # Add to log
        self.feedback_log.append(log_entry)
        
        # Auto-scroll if enabled
        if self.auto_scroll:
            scrollbar = self.feedback_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        # Limit log size
        doc = self.feedback_log.document()
        if doc.blockCount() > self.max_log_entries:
            cursor = self.feedback_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 
                             doc.blockCount() - self.max_log_entries)
            cursor.removeSelectedText()
    
    def start_task(self, task_id: str, description: str, 
                 steps: Optional[List[str]] = None):
        """Start tracking a new task."""
        self.tasks[task_id] = {
            "description": description,
            "progress": 0,
            "steps": steps,
            "current_step": 0 if steps else None,
            "start_time": self._get_timestamp(),
            "status": "Running"
        }
        
        self.activity_label.setText(f"Task: {description}")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        self.add_info(f"Started task: {description}")
    
    def update_task_progress(self, task_id: str, progress: int, 
                           step_description: Optional[str] = None):
        """Update the progress of a task."""
        if task_id not in self.tasks:
            self.add_warning(f"Attempted to update unknown task: {task_id}")
            return
            
        task = self.tasks[task_id]
        task["progress"] = progress
        
        if step_description:
            task["current_step_description"] = step_description
            self.add_info(f"Task step: {step_description}")
            
        self.progress_bar.setValue(progress)
        
        if progress == 100:
            self.complete_task(task_id)
    
    def advance_task_step(self, task_id: str):
        """Advance to the next step in a task."""
        if task_id not in self.tasks or not self.tasks[task_id].get("steps"):
            return
            
        task = self.tasks[task_id]
        task["current_step"] += 1
        
        if task["current_step"] < len(task["steps"]):
            step = task["steps"][task["current_step"]]
            task["current_step_description"] = step
            self.add_info(f"Task step: {step}")
            
            # Update progress
            progress = int((task["current_step"] / len(task["steps"])) * 100)
            task["progress"] = progress
            self.progress_bar.setValue(progress)
    
    def complete_task(self, task_id: str, success: bool = True, 
                    message: Optional[str] = None):
        """Mark a task as complete."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task["end_time"] = self._get_timestamp()
        task["status"] = "Completed" if success else "Failed"
        
        if success:
            self.add_info(f"Task completed: {task['description']}" + 
                        (f" - {message}" if message else ""))
        else:
            self.add_warning(f"Task failed: {task['description']}" + 
                           (f" - {message}" if message else ""))
            
        self.progress_bar.setVisible(False)
        self.activity_label.setText("No active tasks")
    
    def update_screen_visualization(self, image_data: bytes, 
                                  objects: Optional[List[Dict[str, Any]]] = None):
        """Update the screen visualization with a new screenshot and recognized objects."""
        # Convert image data to QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        
        # Scale the image to fit the view while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.screen_view.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # If we have objects to highlight and highlighting is enabled
        if objects and self.highlight_objects_checkbox.isChecked():
            # Create a copy of the pixmap that we can draw on
            image = scaled_pixmap.toImage()
            
            # Scale factor for object coordinates
            scale_x = scaled_pixmap.width() / pixmap.width()
            scale_y = scaled_pixmap.height() / pixmap.height()
            
            # Draw object boundaries (simplified - would need proper painting in real implementation)
            # In a real implementation, you would create a QPainter and draw rectangles
            pass
        
        # Set the pixmap
        self.screen_view.setPixmap(scaled_pixmap)
        
        # Update tab to show screen tab
        self.tab_widget.setCurrentIndex(1)  # Screen visualization tab
    
    def _refresh_screen_view(self):
        """Refresh the screen visualization."""
        # This would typically request a new screenshot from the agent
        self.add_info("Refreshing screen visualization...")
    
    def _on_highlight_toggled(self, checked: bool):
        """Handle toggling of object highlighting."""
        self.add_info(f"Object highlighting {'enabled' if checked else 'disabled'}")
        # Would typically trigger a refresh of the view with/without highlighting
    
    def _on_verbosity_changed(self, index: int):
        """Handle changes to verbosity level."""
        level_name = self.verbosity_selector.currentText().upper()
        self.verbosity_level = VerbosityLevel[level_name]
        
        self.add_info(f"Verbosity level changed to {level_name}")
        
        # Emit signal for preferences change
        self.preferences_changed.emit({"verbosity_level": level_name})
    
    def _on_timestamp_toggled(self, checked: bool):
        """Handle toggling of timestamp display."""
        self.show_timestamps = checked
        self.add_info(f"Timestamps {'shown' if checked else 'hidden'}")
        
        # Emit signal for preferences change
        self.preferences_changed.emit({"show_timestamps": checked})
    
    def _on_autoscroll_toggled(self, checked: bool):
        """Handle toggling of auto-scroll."""
        self.auto_scroll = checked
        self.add_info(f"Auto-scroll {'enabled' if checked else 'disabled'}")
        
        # Emit signal for preferences change
        self.preferences_changed.emit({"auto_scroll": checked})
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def clear_log(self):
        """Clear the feedback log."""
        self.feedback_log.clear()
        
    def get_log_content(self) -> str:
        """Get the content of the feedback log."""
        return self.feedback_log.toPlainText()

    def add_diagnostic(self, title: str, diagnostic_data: Dict[str, Any]) -> None:
        """
        Add diagnostic information to the feedback display.
        
        Args:
            title: Title for the diagnostic information
            diagnostic_data: Dictionary containing diagnostic data
        """
        # Format diagnostic data as HTML
        html = f'<div class="diagnostic"><h3>📊 {title}</h3><ul>'
        
        for key, value in diagnostic_data.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                html += f'<li><b>{key}:</b><ul>'
                for sub_key, sub_value in value.items():
                    html += f'<li><b>{sub_key}:</b> {sub_value}</li>'
                html += '</ul></li>'
            elif isinstance(value, list):
                # Handle lists
                html += f'<li><b>{key}:</b><ul>'
                for item in value:
                    html += f'<li>{item}</li>'
                html += '</ul></li>'
            else:
                # Handle simple values
                html += f'<li><b>{key}:</b> {value}</li>'
                
        html += '</ul></div>'
        
        # Apply style based on verbosity
        if self.verbosity_level in [VerbosityLevel.DETAILED, VerbosityLevel.DEBUG]:
            self.feedback_log.append(html)
            # Scroll to the bottom
            self.feedback_log.verticalScrollBar().setValue(
                self.feedback_log.verticalScrollBar().maximum()
            )
        
        # Even if not shown, store in history
        self.history.append({
            "type": "diagnostic",
            "title": title,
            "data": diagnostic_data,
            "timestamp": time.time()
        }) 
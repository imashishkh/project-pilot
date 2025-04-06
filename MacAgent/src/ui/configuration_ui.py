"""
ConfigurationUI Module

This module provides the configuration interface for MacAgent.
"""

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QLabel, QPushButton, QComboBox, QCheckBox,
                           QLineEdit, QSpinBox, QFormLayout, QGroupBox,
                           QScrollArea, QSlider, QColorDialog, QMessageBox,
                           QFileDialog, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from MacAgent.src.utils.settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class ConfigurationUI(QWidget):
    """
    A user interface for configuring the AI agent settings.
    
    Features:
    - Adjustment of agent parameters
    - Profiles for different use cases
    - Permission management
    - System status monitoring
    - Theme customization
    """
    
    # Signal when configuration changes
    config_changed = pyqtSignal(dict)
    
    def __init__(self, settings_manager: Optional[SettingsManager] = None, parent=None):
        """Initialize the configuration UI."""
        super().__init__(parent)
        
        # Initialize settings manager
        self.settings_manager = settings_manager or SettingsManager()
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("MacAgent Configuration")
        title_label.setFont(QFont("Helvetica", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Profiles selection
        profile_layout = QHBoxLayout()
        profile_label = QLabel("Profile:")
        self.profile_selector = QComboBox()
        self.profile_selector.addItems(self.settings_manager.get_profiles())
        self.profile_selector.setCurrentText(self.settings_manager.get_current_profile())
        self.profile_selector.currentTextChanged.connect(self._on_profile_changed)
        
        self.new_profile_button = QPushButton("New")
        self.new_profile_button.clicked.connect(self._on_new_profile)
        
        self.delete_profile_button = QPushButton("Delete")
        self.delete_profile_button.clicked.connect(self._on_delete_profile)
        
        profile_layout.addWidget(profile_label)
        profile_layout.addWidget(self.profile_selector)
        profile_layout.addWidget(self.new_profile_button)
        profile_layout.addWidget(self.delete_profile_button)
        profile_layout.addStretch()
        
        main_layout.addLayout(profile_layout)
        
        # Create tab widget for different configuration sections
        self.tab_widget = QTabWidget()
        
        # General settings tab
        general_tab = self._create_general_tab()
        self.tab_widget.addTab(general_tab, "General")
        
        # Appearance tab
        appearance_tab = self._create_appearance_tab()
        self.tab_widget.addTab(appearance_tab, "Appearance")
        
        # System tab
        system_tab = self._create_system_tab()
        self.tab_widget.addTab(system_tab, "System")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Save and Cancel buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._on_save_config)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_changes)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Load current profile settings
        self._load_profile_settings()
        
        # Start system monitoring
        self._update_system_stats()
        
    def _create_general_tab(self) -> QWidget:
        """Create the general settings tab."""
        general_tab = QScrollArea()
        general_tab.setWidgetResizable(True)
        general_widget = QWidget()
        general_layout = QVBoxLayout(general_widget)
        
        # Agent parameters
        parameters_group = QGroupBox("Agent Parameters")
        parameters_layout = QFormLayout(parameters_group)
        
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(["GPT-4", "GPT-3.5", "Claude", "Local Model"])
        parameters_layout.addRow("AI Model:", self.model_selector)
        
        # Response verbosity
        self.verbosity_slider = QSlider(Qt.Horizontal)
        self.verbosity_slider.setRange(1, 4)
        self.verbosity_slider.setValue(2)
        self.verbosity_slider.setTickPosition(QSlider.TicksBelow)
        parameters_layout.addRow("Response Verbosity:", self.verbosity_slider)
        
        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        parameters_layout.addRow("API Key:", self.api_key_input)
        
        general_layout.addWidget(parameters_group)
        
        # Permissions group
        permissions_group = QGroupBox("Permissions")
        permissions_layout = QVBoxLayout(permissions_group)
        
        # Screen recording permission
        self.screen_recording_checkbox = QCheckBox("Allow Screen Recording")
        self.screen_recording_checkbox.setChecked(True)
        permissions_layout.addWidget(self.screen_recording_checkbox)
        
        # File system access
        self.file_access_checkbox = QCheckBox("Allow File System Access")
        self.file_access_checkbox.setChecked(True)
        permissions_layout.addWidget(self.file_access_checkbox)
        
        # Automation control
        self.automation_checkbox = QCheckBox("Allow UI Automation")
        self.automation_checkbox.setChecked(True)
        permissions_layout.addWidget(self.automation_checkbox)
        
        # Internet access
        self.internet_checkbox = QCheckBox("Allow Internet Access")
        self.internet_checkbox.setChecked(True)
        permissions_layout.addWidget(self.internet_checkbox)
        
        general_layout.addWidget(permissions_group)
        general_layout.addStretch()
        
        general_tab.setWidget(general_widget)
        return general_tab
    
    def _create_appearance_tab(self) -> QWidget:
        """Create the appearance settings tab."""
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        
        # Theme selection
        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["System Default", "Light", "Dark", "Custom"])
        theme_layout.addRow("Theme:", self.theme_selector)
        
        # Custom colors
        self.primary_color_button = QPushButton("Select")
        self.primary_color_button.clicked.connect(
            lambda: self._on_color_select("primary_color"))
        theme_layout.addRow("Primary Color:", self.primary_color_button)
        
        appearance_layout.addWidget(theme_group)
        
        # Font settings
        font_group = QGroupBox("Font Settings")
        font_layout = QFormLayout(font_group)
        
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 24)
        self.font_size_spinbox.setValue(12)
        font_layout.addRow("Font Size:", self.font_size_spinbox)
        
        appearance_layout.addWidget(font_group)
        appearance_layout.addStretch()
        
        return appearance_tab
    
    def _create_system_tab(self) -> QWidget:
        """Create the system settings tab."""
        system_tab = QWidget()
        system_layout = QVBoxLayout(system_tab)
        
        # System status
        status_group = QGroupBox("System Status")
        status_layout = QFormLayout(status_group)
        
        self.memory_usage_label = QLabel("0%")
        status_layout.addRow("Memory Usage:", self.memory_usage_label)
        
        self.cpu_usage_label = QLabel("0%")
        status_layout.addRow("CPU Usage:", self.cpu_usage_label)
        
        system_layout.addWidget(status_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.reset_button = QPushButton("Reset All Settings")
        self.reset_button.clicked.connect(self._on_reset_settings)
        actions_layout.addWidget(self.reset_button)
        
        self.export_button = QPushButton("Export Settings")
        self.export_button.clicked.connect(self._on_export_settings)
        actions_layout.addWidget(self.export_button)
        
        self.import_button = QPushButton("Import Settings")
        self.import_button.clicked.connect(self._on_import_settings)
        actions_layout.addWidget(self.import_button)
        
        system_layout.addWidget(actions_group)
        system_layout.addStretch()
        
        return system_tab
    
    def _load_profile_settings(self):
        """Load the current profile settings into the UI."""
        config = self.settings_manager.get_current_config()
        
        # Update UI elements with profile settings
        self.model_selector.setCurrentText(config.get("model", "GPT-4"))
        self.verbosity_slider.setValue(config.get("verbosity", 2))
        self.api_key_input.setText(config.get("api_key", ""))
        
        permissions = config.get("permissions", {})
        self.screen_recording_checkbox.setChecked(permissions.get("screen_recording", True))
        self.file_access_checkbox.setChecked(permissions.get("file_access", True))
        self.automation_checkbox.setChecked(permissions.get("automation", True))
        self.internet_checkbox.setChecked(permissions.get("internet", True))
        
        appearance = config.get("appearance", {})
        self.theme_selector.setCurrentText(appearance.get("theme", "System Default"))
        self.font_size_spinbox.setValue(appearance.get("font_size", 12))
        
        # Update button background with primary color
        primary_color = appearance.get("primary_color", "#0078D7")
        self.primary_color_button.setStyleSheet(f"background-color: {primary_color}")
    
    def _save_profile_settings(self):
        """Save the current UI settings to the profile."""
        # Gather settings from UI elements
        config = {
            "model": self.model_selector.currentText(),
            "verbosity": self.verbosity_slider.value(),
            "api_key": self.api_key_input.text(),
            "permissions": {
                "screen_recording": self.screen_recording_checkbox.isChecked(),
                "file_access": self.file_access_checkbox.isChecked(),
                "automation": self.automation_checkbox.isChecked(),
                "internet": self.internet_checkbox.isChecked()
            },
            "appearance": {
                "theme": self.theme_selector.currentText(),
                "primary_color": self.settings_manager.get_setting("appearance.primary_color", "#0078D7"),
                "font_size": self.font_size_spinbox.value()
            }
        }
        
        # Update settings in manager
        self.settings_manager.update_settings(config)
    
    def _on_profile_changed(self, profile_name: str):
        """Handle profile selection change."""
        current_profile = self.settings_manager.get_current_profile()
        if profile_name != current_profile:
            # Switch to selected profile
            self.settings_manager.switch_profile(profile_name)
            self._load_profile_settings()
    
    def _on_new_profile(self):
        """Create a new profile."""
        profile_name, ok = QInputDialog.getText(self, "New Profile", "Enter profile name:")
        
        if ok and profile_name:
            profiles = self.settings_manager.get_profiles()
            if profile_name in profiles:
                QMessageBox.warning(self, "Profile Exists", f"A profile named '{profile_name}' already exists.")
                return
                
            # Create new profile
            self.settings_manager.create_profile(profile_name)
            
            # Update UI
            self.profile_selector.addItem(profile_name)
            self.profile_selector.setCurrentText(profile_name)
    
    def _on_delete_profile(self):
        """Delete the current profile."""
        profile_name = self.settings_manager.get_current_profile()
        
        if profile_name == "Default":
            QMessageBox.warning(self, "Cannot Delete", "The Default profile cannot be deleted.")
            return
            
        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the profile '{profile_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Delete profile
            if self.settings_manager.delete_profile(profile_name):
                # Update UI
                index = self.profile_selector.findText(profile_name)
                self.profile_selector.removeItem(index)
                self._load_profile_settings()
    
    def _on_color_select(self, color_type: str):
        """Open color dialog for selecting colors."""
        current_color = QColor(self.settings_manager.get_setting(f"appearance.{color_type}", "#0078D7"))
        
        color = QColorDialog.getColor(current_color, self, f"Select {color_type.replace('_', ' ').title()}")
        
        if color.isValid():
            # Update the setting
            self.settings_manager.set_setting(f"appearance.{color_type}", color.name())
            
            # Update button background as visual feedback
            self.primary_color_button.setStyleSheet(f"background-color: {color.name()}")
    
    def _on_save_config(self):
        """Save the configuration."""
        self._save_profile_settings()
        if self.settings_manager.save_settings():
            # Emit signal for configuration change
            self.config_changed.emit(self.settings_manager.get_current_config())
            QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
        else:
            QMessageBox.warning(self, "Save Error", "Failed to save settings.")
    
    def _on_cancel_changes(self):
        """Cancel changes and reload from saved config."""
        self.settings_manager.load_settings()
        self._load_profile_settings()
        QMessageBox.information(self, "Changes Canceled", "All changes have been discarded.")
    
    def _on_reset_settings(self):
        """Reset all settings to default."""
        confirm = QMessageBox.question(
            self,
            "Confirm Reset",
            "Are you sure you want to reset all settings to default?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            self.settings_manager.reset_to_defaults()
            self._load_profile_settings()
    
    def _on_export_settings(self):
        """Export settings to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Settings",
            os.path.expanduser("~/Desktop/macagent_settings.json"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            if self.settings_manager.export_settings(file_path):
                QMessageBox.information(self, "Export Successful", f"Settings exported to {file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export settings.")
    
    def _on_import_settings(self):
        """Import settings from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Settings",
            os.path.expanduser("~/Desktop"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            if self.settings_manager.import_settings(file_path):
                # Update profile selector with new profiles
                self.profile_selector.clear()
                self.profile_selector.addItems(self.settings_manager.get_profiles())
                self.profile_selector.setCurrentText(self.settings_manager.get_current_profile())
                self._load_profile_settings()
                
                QMessageBox.information(
                    self, 
                    "Import Successful", 
                    f"Imported {len(self.settings_manager.get_profiles())} profiles"
                )
            else:
                QMessageBox.warning(self, "Import Failed", "Failed to import settings.")
    
    def _update_system_stats(self):
        """Update system statistics."""
        try:
            import psutil
            self.memory_usage_label.setText(f"{psutil.virtual_memory().percent}%")
            self.cpu_usage_label.setText(f"{psutil.cpu_percent()}%")
        except ImportError:
            self.memory_usage_label.setText("N/A")
            self.cpu_usage_label.setText("N/A")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.settings_manager.get_current_config() 
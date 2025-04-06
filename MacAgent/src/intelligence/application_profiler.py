"""
Application Profiler Module

This module provides functionality for analyzing, learning, and maintaining
profiles of macOS applications for enhanced AI agent interaction.
"""

import os
import json
import time
import logging
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import Cocoa
from AppKit import NSWorkspace, NSRunningApplication, NSApplicationActivationPolicy

logger = logging.getLogger(__name__)


class ApplicationProfile:
    """Represents a profile for a specific application."""
    
    def __init__(
        self,
        bundle_id: str,
        name: str,
        version: str = "",
        path: str = "",
        last_updated: Optional[datetime] = None
    ):
        """Initialize an application profile.
        
        Args:
            bundle_id: Application bundle identifier
            name: Application name
            version: Application version
            path: Path to the application bundle
            last_updated: When the profile was last updated
        """
        self.bundle_id = bundle_id
        self.name = name
        self.version = version
        self.path = path
        self.last_updated = last_updated or datetime.now()
        
        # UI element patterns observed in the application
        self.ui_elements: Dict[str, Dict[str, Any]] = {}
        
        # Common actions and their success rates
        self.actions: Dict[str, Dict[str, float]] = {}
        
        # Application preferences and settings
        self.preferences: Dict[str, Any] = {}
        
        # Usage statistics
        self.launch_count: int = 0
        self.total_usage_time: int = 0  # in seconds
        self.last_used: Optional[datetime] = None
        
        # Accessibility information
        self.accessibility_features: Dict[str, bool] = {
            "supports_accessibility": False,
            "needs_screen_recording": False,
            "has_keyboard_shortcuts": False
        }
        
        # Special handling flags
        self.quirks: Dict[str, Any] = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationProfile':
        """Create an ApplicationProfile from a dictionary.
        
        Args:
            data: Dictionary containing profile data
            
        Returns:
            New ApplicationProfile instance
        """
        profile = cls(
            bundle_id=data.get("bundle_id", ""),
            name=data.get("name", ""),
            version=data.get("version", ""),
            path=data.get("path", ""),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        )
        
        profile.ui_elements = data.get("ui_elements", {})
        profile.actions = data.get("actions", {})
        profile.preferences = data.get("preferences", {})
        profile.launch_count = data.get("launch_count", 0)
        profile.total_usage_time = data.get("total_usage_time", 0)
        
        if "last_used" in data and data["last_used"]:
            profile.last_used = datetime.fromisoformat(data["last_used"])
        
        profile.accessibility_features = data.get("accessibility_features", {
            "supports_accessibility": False,
            "needs_screen_recording": False,
            "has_keyboard_shortcuts": False
        })
        
        profile.quirks = data.get("quirks", {})
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary.
        
        Returns:
            Dictionary representation of the profile
        """
        data = {
            "bundle_id": self.bundle_id,
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "last_updated": self.last_updated.isoformat(),
            "ui_elements": self.ui_elements,
            "actions": self.actions,
            "preferences": self.preferences,
            "launch_count": self.launch_count,
            "total_usage_time": self.total_usage_time,
            "accessibility_features": self.accessibility_features,
            "quirks": self.quirks
        }
        
        if self.last_used:
            data["last_used"] = self.last_used.isoformat()
        
        return data
    
    def update_usage_stats(self, usage_time: int = 0) -> None:
        """Update usage statistics for the application.
        
        Args:
            usage_time: Time spent using the application in seconds
        """
        self.launch_count += 1
        self.total_usage_time += usage_time
        self.last_used = datetime.now()
    
    def record_action_result(self, action: str, success: bool) -> None:
        """Record the result of an action for learning.
        
        Args:
            action: The action performed
            success: Whether the action was successful
        """
        if action not in self.actions:
            self.actions[action] = {
                "success_count": 0,
                "attempt_count": 0,
                "success_rate": 0.0
            }
        
        self.actions[action]["attempt_count"] += 1
        if success:
            self.actions[action]["success_count"] += 1
        
        # Update success rate
        attempts = self.actions[action]["attempt_count"]
        successes = self.actions[action]["success_count"]
        self.actions[action]["success_rate"] = successes / attempts if attempts > 0 else 0
    
    def add_ui_element(self, element_id: str, properties: Dict[str, Any]) -> None:
        """Add or update a UI element in the profile.
        
        Args:
            element_id: Unique identifier for the element
            properties: Properties of the element
        """
        self.ui_elements[element_id] = properties
        self.last_updated = datetime.now()
    
    def get_action_success_rate(self, action: str) -> float:
        """Get the success rate for a specific action.
        
        Args:
            action: The action to check
            
        Returns:
            Success rate as a float between 0 and 1
        """
        if action not in self.actions:
            return 0.0
        
        return self.actions[action].get("success_rate", 0.0)
    
    def get_usage_frequency_score(self) -> float:
        """Calculate a score representing how frequently this app is used.
        
        Returns:
            Score between 0 and 1, where higher means more frequently used
        """
        # Factor in recency
        recency_score = 0.0
        if self.last_used:
            days_since_use = (datetime.now() - self.last_used).days
            recency_score = max(0, 1 - (days_since_use / 30))  # Scale down over 30 days
        
        # Factor in frequency
        frequency_score = min(1.0, self.launch_count / 100)  # Cap at 100 launches
        
        # Factor in usage time
        time_score = min(1.0, self.total_usage_time / (3600 * 10))  # Cap at 10 hours
        
        # Combine scores with weights
        return (recency_score * 0.5) + (frequency_score * 0.3) + (time_score * 0.2)
    
    def __str__(self) -> str:
        """String representation of the profile."""
        return f"{self.name} ({self.bundle_id}) - Version: {self.version}, Used: {self.launch_count} times"


class ApplicationProfiler:
    """
    Analyzes and maintains profiles for macOS applications.
    
    Features:
    - Application discovery and analysis
    - UI pattern identification and learning
    - Usage tracking and prioritization
    - Adaptation to application changes
    - Profile persistence and management
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the application profiler.
        
        Args:
            data_dir: Directory for storing profile data, defaults to ~/.macagent/app_profiles
        """
        # Set up data directory
        if data_dir is None:
            self.data_dir = Path.home() / ".macagent" / "app_profiles"
        else:
            self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for profile metadata and usage statistics
        self.db_path = self.data_dir / "app_profiles.db"
        self._init_database()
        
        # In-memory cache of loaded application profiles
        self._profiles: Dict[str, ApplicationProfile] = {}
        
        # Currently running applications
        self._running_apps: Dict[str, Dict[str, Any]] = {}
        
        # Track application usage
        self._app_start_times: Dict[str, float] = {}
        
        # Load system applications on initialization
        self.discover_system_applications()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for profile metadata."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            bundle_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT,
            path TEXT,
            last_updated TEXT,
            launch_count INTEGER DEFAULT 0,
            total_usage_time INTEGER DEFAULT 0,
            last_used TEXT,
            is_system_app INTEGER DEFAULT 0
        )
        ''')
        
        # Create index for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_app_name ON applications(name)')
        
        # Create table for tracking application usage
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bundle_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration INTEGER,
            FOREIGN KEY (bundle_id) REFERENCES applications(bundle_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def discover_system_applications(self) -> List[str]:
        """Discover and profile system applications.
        
        Returns:
            List of bundle IDs for discovered applications
        """
        logger.info("Discovering system applications")
        
        discovered_bundle_ids = []
        
        # Common system application locations
        app_dirs = [
            "/Applications",
            "/System/Applications",
            "/System/Library/CoreServices",
            str(Path.home() / "Applications")
        ]
        
        for app_dir in app_dirs:
            if not os.path.exists(app_dir):
                continue
                
            # Find .app bundles
            for item in os.listdir(app_dir):
                if item.endswith(".app"):
                    app_path = os.path.join(app_dir, item)
                    bundle_id = self._get_bundle_id(app_path)
                    
                    if bundle_id:
                        app_name = item.replace(".app", "")
                        app_version = self._get_app_version(app_path)
                        
                        # Check if we already have this app in our database
                        if not self._is_app_in_database(bundle_id):
                            # Create a new profile
                            profile = ApplicationProfile(
                                bundle_id=bundle_id,
                                name=app_name,
                                version=app_version,
                                path=app_path
                            )
                            
                            # Save the profile
                            self._save_profile_to_db(profile, is_system_app=True)
                            
                            # Store in memory
                            self._profiles[bundle_id] = profile
                            
                            logger.debug(f"Discovered system app: {app_name} ({bundle_id})")
                            discovered_bundle_ids.append(bundle_id)
        
        logger.info(f"Discovered {len(discovered_bundle_ids)} system applications")
        return discovered_bundle_ids
    
    def _get_bundle_id(self, app_path: str) -> Optional[str]:
        """Get the bundle ID for an application.
        
        Args:
            app_path: Path to the application bundle
            
        Returns:
            Bundle ID or None if not found
        """
        plist_path = os.path.join(app_path, "Contents", "Info.plist")
        
        if not os.path.exists(plist_path):
            return None
        
        try:
            # Use PlistBuddy to read the bundle ID
            cmd = ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleIdentifier", plist_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get bundle ID for {app_path}: {e}")
            return None
    
    def _get_app_version(self, app_path: str) -> str:
        """Get the version of an application.
        
        Args:
            app_path: Path to the application bundle
            
        Returns:
            Version string or empty string if not found
        """
        plist_path = os.path.join(app_path, "Contents", "Info.plist")
        
        if not os.path.exists(plist_path):
            return ""
        
        try:
            # Use PlistBuddy to read the version
            cmd = ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleShortVersionString", plist_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to get version for {app_path}: {e}")
            return ""
    
    def _is_app_in_database(self, bundle_id: str) -> bool:
        """Check if an application is in the database.
        
        Args:
            bundle_id: Bundle ID to check
            
        Returns:
            True if the application is in the database, False otherwise
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM applications WHERE bundle_id = ?", (bundle_id,))
        result = cursor.fetchone() is not None
        conn.close()
        return result
    
    def _save_profile_to_db(self, profile: ApplicationProfile, is_system_app: bool = False) -> None:
        """Save application profile metadata to the database.
        
        Args:
            profile: ApplicationProfile to save
            is_system_app: Whether this is a system application
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Insert or update the application record
        cursor.execute('''
        INSERT OR REPLACE INTO applications 
        (bundle_id, name, version, path, last_updated, launch_count, total_usage_time, last_used, is_system_app) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.bundle_id,
            profile.name,
            profile.version,
            profile.path,
            profile.last_updated.isoformat(),
            profile.launch_count,
            profile.total_usage_time,
            profile.last_used.isoformat() if profile.last_used else None,
            1 if is_system_app else 0
        ))
        
        conn.commit()
        conn.close()
        
        # Save the full profile to a JSON file
        self._save_profile_to_file(profile)
    
    def _save_profile_to_file(self, profile: ApplicationProfile) -> None:
        """Save the full application profile to a JSON file.
        
        Args:
            profile: ApplicationProfile to save
        """
        file_path = self.data_dir / f"{profile.bundle_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save profile for {profile.name} to file: {e}")
    
    def _load_profile(self, bundle_id: str) -> Optional[ApplicationProfile]:
        """Load an application profile from storage.
        
        Args:
            bundle_id: Bundle ID of the application
            
        Returns:
            ApplicationProfile if found, None otherwise
        """
        # Check if already in memory
        if bundle_id in self._profiles:
            return self._profiles[bundle_id]
        
        # Try to load from file
        file_path = self.data_dir / f"{bundle_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                profile = ApplicationProfile.from_dict(data)
                
                # Cache in memory
                self._profiles[bundle_id] = profile
                return profile
                
        except Exception as e:
            logger.error(f"Failed to load profile for {bundle_id}: {e}")
            return None
    
    def get_profile(self, bundle_id: str) -> Optional[ApplicationProfile]:
        """Get the profile for an application.
        
        Args:
            bundle_id: Bundle ID of the application
            
        Returns:
            ApplicationProfile if found, None otherwise
        """
        return self._load_profile(bundle_id)
    
    def get_profile_by_name(self, app_name: str) -> Optional[ApplicationProfile]:
        """Get the profile for an application by name.
        
        Args:
            app_name: Name of the application
            
        Returns:
            ApplicationProfile if found, None otherwise
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Search for the application by name
        cursor.execute("SELECT bundle_id FROM applications WHERE name LIKE ?", (f"%{app_name}%",))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        bundle_id = result[0]
        return self._load_profile(bundle_id)
    
    def get_all_profiles(self) -> List[ApplicationProfile]:
        """Get all application profiles.
        
        Returns:
            List of ApplicationProfile objects
        """
        profiles = []
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get all application bundle IDs
        cursor.execute("SELECT bundle_id FROM applications")
        bundle_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Load each profile
        for bundle_id in bundle_ids:
            profile = self._load_profile(bundle_id)
            if profile:
                profiles.append(profile)
        
        return profiles
    
    def get_frequently_used_apps(self, limit: int = 10) -> List[ApplicationProfile]:
        """Get frequently used applications, ordered by usage frequency.
        
        Args:
            limit: Maximum number of applications to return
            
        Returns:
            List of ApplicationProfile objects
        """
        # Load all profiles if not already loaded
        profiles = self.get_all_profiles()
        
        # Sort by usage frequency score
        profiles.sort(key=lambda p: p.get_usage_frequency_score(), reverse=True)
        
        # Return top N profiles
        return profiles[:limit]
    
    def track_app_launch(self, bundle_id: str) -> None:
        """Track the launch of an application.
        
        Args:
            bundle_id: Bundle ID of the application
        """
        self._app_start_times[bundle_id] = time.time()
        
        # Record in the database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO usage_history (bundle_id, start_time) 
        VALUES (?, ?)
        ''', (bundle_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def track_app_quit(self, bundle_id: str) -> None:
        """Track the quit of an application.
        
        Args:
            bundle_id: Bundle ID of the application
        """
        if bundle_id in self._app_start_times:
            # Calculate usage time
            start_time = self._app_start_times.pop(bundle_id)
            usage_time = int(time.time() - start_time)
            
            # Update the profile
            profile = self._load_profile(bundle_id)
            if profile:
                profile.update_usage_stats(usage_time)
                self._save_profile_to_db(profile)
            
            # Update the usage history
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get the most recent unfinished usage record
            cursor.execute('''
            SELECT id FROM usage_history 
            WHERE bundle_id = ? AND end_time IS NULL 
            ORDER BY start_time DESC LIMIT 1
            ''', (bundle_id,))
            
            result = cursor.fetchone()
            if result:
                usage_id = result[0]
                cursor.execute('''
                UPDATE usage_history 
                SET end_time = ?, duration = ? 
                WHERE id = ?
                ''', (datetime.now().isoformat(), usage_time, usage_id))
                
                conn.commit()
            
            conn.close()
    
    def monitor_running_applications(self) -> None:
        """Monitor currently running applications and track their usage."""
        workspace = NSWorkspace.sharedWorkspace()
        running_apps = workspace.runningApplications()
        
        current_bundle_ids = set()
        
        for app in running_apps:
            if app.activationPolicy() == NSApplicationActivationPolicy.Regular:
                bundle_id = app.bundleIdentifier()
                if bundle_id:
                    bundle_id = str(bundle_id)
                    current_bundle_ids.add(bundle_id)
                    
                    # New app started
                    if bundle_id not in self._running_apps:
                        app_name = str(app.localizedName())
                        self._running_apps[bundle_id] = {
                            "name": app_name,
                            "start_time": time.time()
                        }
                        
                        logger.debug(f"App started: {app_name} ({bundle_id})")
                        self.track_app_launch(bundle_id)
                        
                        # Make sure we have a profile
                        if not self._is_app_in_database(bundle_id):
                            app_path = str(app.bundleURL().path())
                            app_version = self._get_app_version(app_path)
                            
                            profile = ApplicationProfile(
                                bundle_id=bundle_id,
                                name=app_name,
                                version=app_version,
                                path=app_path
                            )
                            
                            self._save_profile_to_db(profile)
        
        # Check for apps that have quit
        closed_bundle_ids = set(self._running_apps.keys()) - current_bundle_ids
        for bundle_id in closed_bundle_ids:
            app_info = self._running_apps.pop(bundle_id)
            logger.debug(f"App quit: {app_info['name']} ({bundle_id})")
            self.track_app_quit(bundle_id)
    
    def analyze_application_ui(self, bundle_id: str) -> Dict[str, Any]:
        """Analyze the UI of an application to build patterns.
        
        Args:
            bundle_id: Bundle ID of the application
            
        Returns:
            Dictionary of UI analysis results
        """
        # This would use macOS Accessibility API to analyze the UI
        # For now, we'll return a placeholder
        logger.info(f"Analyzing UI for {bundle_id}")
        
        # Get the profile
        profile = self._load_profile(bundle_id)
        if not profile:
            return {"error": "Profile not found"}
        
        # This would be a comprehensive UI analysis
        # For demonstration, we'll return a placeholder
        results = {
            "analyzed": True,
            "elements_found": 0,
            "common_patterns": [],
            "accessibility_score": 0
        }
        
        # Update the profile with our findings
        profile.last_updated = datetime.now()
        self._save_profile_to_db(profile)
        
        return results
    
    def identify_common_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Identify common UI patterns across applications.
        
        Returns:
            Dictionary of pattern categories and their instances
        """
        # This would analyze patterns across all known applications
        # For demonstration, we'll return a placeholder
        logger.info("Identifying common patterns across applications")
        
        # Get all profiles
        profiles = self.get_all_profiles()
        
        patterns = {
            "menus": [],
            "dialogs": [],
            "buttons": [],
            "text_fields": [],
            "toolbars": []
        }
        
        # In a real implementation, this would do pattern recognition
        return patterns
    
    def detect_application_changes(self, bundle_id: str) -> Dict[str, Any]:
        """Detect changes in an application compared to its profile.
        
        Args:
            bundle_id: Bundle ID of the application
            
        Returns:
            Dictionary of detected changes
        """
        # Get the profile
        profile = self._load_profile(bundle_id)
        if not profile:
            return {"error": "Profile not found"}
        
        # Get the current version
        app_path = profile.path
        current_version = self._get_app_version(app_path)
        
        changes = {
            "version_changed": current_version != profile.version,
            "old_version": profile.version,
            "new_version": current_version,
            "ui_changes": []
        }
        
        if changes["version_changed"]:
            logger.info(f"Application {profile.name} updated from {profile.version} to {current_version}")
            
            # Update the profile version
            profile.version = current_version
            profile.last_updated = datetime.now()
            self._save_profile_to_db(profile)
            
            # This would trigger a re-analysis of the UI
            # For demonstration, we just return the version change
        
        return changes
    
    def record_interaction_result(self, bundle_id: str, action: str, success: bool) -> None:
        """Record the result of an interaction with an application.
        
        Args:
            bundle_id: Bundle ID of the application
            action: The action performed
            success: Whether the action was successful
        """
        profile = self._load_profile(bundle_id)
        if profile:
            profile.record_action_result(action, success)
            self._save_profile_to_db(profile)
    
    def get_application_suggestions(self, task: str) -> List[Dict[str, Any]]:
        """Get application suggestions for a specific task.
        
        Args:
            task: Description of the task
            
        Returns:
            List of suggested applications with their scores
        """
        # This would use NLP to match task to application capabilities
        # For demonstration, we'll return a simple keyword match
        logger.info(f"Finding application suggestions for task: {task}")
        
        suggestions = []
        
        # Get all profiles
        profiles = self.get_all_profiles()
        
        # Simple keyword matching (in a real implementation, this would be NLP-based)
        keywords = {
            "write": ["Pages", "TextEdit", "Microsoft Word"],
            "email": ["Mail", "Microsoft Outlook"],
            "browse": ["Safari", "Chrome", "Firefox"],
            "photo": ["Photos", "Photoshop", "Lightroom"],
            "video": ["iMovie", "Final Cut Pro", "QuickTime Player"],
            "terminal": ["Terminal", "iTerm"],
            "code": ["Visual Studio Code", "Xcode", "Sublime Text"]
        }
        
        matched_apps = set()
        for keyword, apps in keywords.items():
            if keyword.lower() in task.lower():
                matched_apps.update(apps)
        
        # Score matching applications based on usage frequency
        for profile in profiles:
            app_score = 0.0
            
            # Base score from usage frequency
            usage_score = profile.get_usage_frequency_score()
            
            # Bonus for matching keywords
            if profile.name in matched_apps:
                app_score = 0.7 + (usage_score * 0.3)  # 70% match, 30% usage
            else:
                app_score = usage_score * 0.3  # Only usage-based
            
            if app_score > 0:
                suggestions.append({
                    "name": profile.name,
                    "bundle_id": profile.bundle_id,
                    "score": app_score,
                    "reason": "Matched task keywords" if profile.name in matched_apps else "Frequently used"
                })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:5]  # Return top 5
    
    def cleanup_old_profiles(self, days_threshold: int = 90) -> int:
        """Cleanup profiles for applications that haven't been used recently.
        
        Args:
            days_threshold: Number of days of inactivity before cleanup
            
        Returns:
            Number of profiles removed
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Find applications that haven't been used in a while
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        cursor.execute('''
        SELECT bundle_id FROM applications 
        WHERE last_used < ? AND is_system_app = 0
        ''', (threshold_date,))
        
        bundle_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete from database
        cursor.execute('''
        DELETE FROM applications 
        WHERE bundle_id IN ({})
        '''.format(','.join(['?'] * len(bundle_ids))), bundle_ids)
        
        conn.commit()
        conn.close()
        
        # Delete profile files
        for bundle_id in bundle_ids:
            file_path = self.data_dir / f"{bundle_id}.json"
            if file_path.exists():
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete profile file for {bundle_id}: {e}")
            
            # Remove from memory cache
            if bundle_id in self._profiles:
                del self._profiles[bundle_id]
        
        logger.info(f"Cleaned up {len(bundle_ids)} old application profiles")
        return len(bundle_ids)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    profiler = ApplicationProfiler()
    
    # Discover system applications
    profiler.discover_system_applications()
    
    # Monitor running applications
    profiler.monitor_running_applications()
    
    # Get frequently used applications
    freq_apps = profiler.get_frequently_used_apps(5)
    print("Frequently used applications:")
    for app in freq_apps:
        print(f"- {app.name} (Score: {app.get_usage_frequency_score():.2f})")
    
    # Get application suggestions for a task
    suggestions = profiler.get_application_suggestions("write a document")
    print("\nSuggested applications for writing a document:")
    for suggestion in suggestions:
        print(f"- {suggestion['name']} (Score: {suggestion['score']:.2f}, Reason: {suggestion['reason']})")

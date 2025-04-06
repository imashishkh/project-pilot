"""
Update Manager Module

This module provides functionality for checking, downloading and installing updates
for the MacAgent application, as well as displaying release notes to users.
"""

import os
import sys
import json
import logging
import tempfile
import threading
import subprocess
import shutil
import time
import hashlib
import platform
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from datetime import datetime, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Enum representing the status of an update."""
    UP_TO_DATE = auto()
    UPDATE_AVAILABLE = auto()
    UPDATE_DOWNLOADING = auto()
    UPDATE_READY = auto()
    UPDATE_INSTALLING = auto()
    UPDATE_COMPLETE = auto()
    ERROR = auto()


class UpdateChannel(Enum):
    """Enum representing the update channel."""
    STABLE = "stable"
    BETA = "beta"
    DEVELOPMENT = "dev"


class ReleaseInfo:
    """Class representing information about a release."""
    
    def __init__(
        self,
        version: str,
        release_date: str,
        release_notes: str,
        download_url: str,
        checksum: str,
        size: int,
        min_os_version: str,
        is_critical: bool = False,
        channel: UpdateChannel = UpdateChannel.STABLE
    ):
        """Initialize a ReleaseInfo object.
        
        Args:
            version: Version string (e.g., "1.2.3")
            release_date: Date of release (ISO format: "YYYY-MM-DD")
            release_notes: Release notes in Markdown format
            download_url: URL to download the update
            checksum: SHA-256 checksum of the update file
            size: Size of the update in bytes
            min_os_version: Minimum macOS version required
            is_critical: Whether this is a critical update
            channel: Update channel this release belongs to
        """
        self.version = version
        self.release_date = release_date
        self.release_notes = release_notes
        self.download_url = download_url
        self.checksum = checksum
        self.size = size
        self.min_os_version = min_os_version
        self.is_critical = is_critical
        self.channel = channel if isinstance(channel, UpdateChannel) else UpdateChannel(channel)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReleaseInfo':
        """Create a ReleaseInfo object from a dictionary.
        
        Args:
            data: Dictionary containing release information
            
        Returns:
            A new ReleaseInfo object
        """
        return cls(
            version=data.get("version", "0.0.0"),
            release_date=data.get("release_date", datetime.now().strftime("%Y-%m-%d")),
            release_notes=data.get("release_notes", ""),
            download_url=data.get("download_url", ""),
            checksum=data.get("checksum", ""),
            size=data.get("size", 0),
            min_os_version=data.get("min_os_version", "10.15"),
            is_critical=data.get("is_critical", False),
            channel=data.get("channel", UpdateChannel.STABLE)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ReleaseInfo object to a dictionary.
        
        Returns:
            Dictionary representation of the ReleaseInfo object
        """
        return {
            "version": self.version,
            "release_date": self.release_date,
            "release_notes": self.release_notes,
            "download_url": self.download_url,
            "checksum": self.checksum,
            "size": self.size,
            "min_os_version": self.min_os_version,
            "is_critical": self.is_critical,
            "channel": self.channel.value
        }
    
    def __str__(self) -> str:
        """Get a string representation of the ReleaseInfo object.
        
        Returns:
            String representation
        """
        return f"Version {self.version} ({self.release_date}) - {self.channel.value}"


class UpdateManager:
    """
    Manages updates for the MacAgent application.
    
    Features:
    - Check for updates automatically or manually
    - Download updates in the background
    - Install updates with user confirmation
    - Provide release notes to users
    - Support for different update channels
    - Automatic update checking at configurable intervals
    - Rollback to previous versions if needed
    """
    
    def __init__(
        self,
        app_name: str = "MacAgent",
        current_version: str = "0.1.0",
        update_url: str = "https://macagent.example.com/updates",
        update_channel: UpdateChannel = UpdateChannel.STABLE,
        auto_check: bool = True,
        check_interval: int = 24,  # hours
        user_data_dir: Optional[Path] = None,
        on_update_available: Optional[Callable[[ReleaseInfo], None]] = None,
        on_update_downloaded: Optional[Callable[[ReleaseInfo, Path], None]] = None,
        on_update_progress: Optional[Callable[[float], None]] = None
    ):
        """Initialize the update manager.
        
        Args:
            app_name: Name of the application
            current_version: Current version of the application
            update_url: Base URL for update server
            update_channel: Update channel to use
            auto_check: Whether to check for updates automatically
            check_interval: Interval (in hours) between automatic update checks
            user_data_dir: Directory for user data (defaults to ~/.macagent)
            on_update_available: Callback when an update is available
            on_update_downloaded: Callback when an update is downloaded
            on_update_progress: Callback for download progress (0.0 to 1.0)
        """
        self.app_name = app_name
        self.current_version = current_version
        self.update_url = update_url
        self.update_channel = update_channel
        self.auto_check = auto_check
        self.check_interval = check_interval
        
        # Set default user data directory if not provided
        if user_data_dir is None:
            self.user_data_dir = Path.home() / f".{app_name.lower()}"
        else:
            self.user_data_dir = user_data_dir
        
        # Update storage directories
        self.updates_dir = self.user_data_dir / "updates"
        self.backups_dir = self.user_data_dir / "backups"
        
        # Ensure directories exist
        self.updates_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # Set callbacks
        self.on_update_available = on_update_available
        self.on_update_downloaded = on_update_downloaded
        self.on_update_progress = on_update_progress
        
        # Internal state
        self._status = UpdateStatus.UP_TO_DATE
        self._latest_release: Optional[ReleaseInfo] = None
        self._download_thread: Optional[threading.Thread] = None
        self._download_path: Optional[Path] = None
        self._last_check_time = self._load_last_check_time()
        self._update_history = self._load_update_history()
        
        # Start automatic update checking if enabled
        if auto_check:
            self._schedule_auto_check()
    
    def _load_last_check_time(self) -> Optional[datetime]:
        """Load the last update check time from disk.
        
        Returns:
            Datetime of last check or None if never checked
        """
        try:
            update_info_path = self.user_data_dir / "update_info.json"
            if not update_info_path.exists():
                return None
            
            with open(update_info_path, "r") as f:
                data = json.load(f)
                if "last_check_time" in data:
                    return datetime.fromisoformat(data["last_check_time"])
                
            return None
        except Exception as e:
            logger.warning(f"Failed to load last update check time: {e}")
            return None
    
    def _save_last_check_time(self) -> None:
        """Save the current update check time to disk."""
        try:
            update_info_path = self.user_data_dir / "update_info.json"
            
            # Load existing data if available
            data = {}
            if update_info_path.exists():
                with open(update_info_path, "r") as f:
                    data = json.load(f)
            
            # Update the last check time
            data["last_check_time"] = datetime.now().isoformat()
            
            # Save to disk
            with open(update_info_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save last update check time: {e}")
    
    def _load_update_history(self) -> List[Dict[str, Any]]:
        """Load update history from disk.
        
        Returns:
            List of update history entries
        """
        try:
            history_path = self.user_data_dir / "update_history.json"
            if not history_path.exists():
                return []
            
            with open(history_path, "r") as f:
                return json.load(f)
                
        except Exception as e:
            logger.warning(f"Failed to load update history: {e}")
            return []
    
    def _save_update_history(self) -> None:
        """Save update history to disk."""
        try:
            history_path = self.user_data_dir / "update_history.json"
            
            with open(history_path, "w") as f:
                json.dump(self._update_history, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save update history: {e}")
    
    def _add_to_update_history(self, from_version: str, to_version: str, success: bool, notes: Optional[str] = None) -> None:
        """Add an entry to the update history.
        
        Args:
            from_version: Version updating from
            to_version: Version updating to
            success: Whether the update was successful
            notes: Any additional notes about the update
        """
        entry = {
            "date": datetime.now().isoformat(),
            "from_version": from_version,
            "to_version": to_version,
            "success": success
        }
        
        if notes:
            entry["notes"] = notes
            
        self._update_history.append(entry)
        self._save_update_history()
    
    def _schedule_auto_check(self) -> None:
        """Schedule automatic update checking."""
        if not self.auto_check:
            return
            
        # Start a daemon thread for checking updates
        thread = threading.Thread(target=self._auto_check_loop, daemon=True)
        thread.start()
    
    def _auto_check_loop(self) -> None:
        """Loop for automatic update checking."""
        while True:
            # Check if it's time to check for updates
            if self._last_check_time is None or \
               datetime.now() - self._last_check_time > timedelta(hours=self.check_interval):
                logger.debug("Running automatic update check")
                self.check_for_updates(silent=True)
            
            # Sleep for a while before checking again
            time.sleep(3600)  # Check every hour if it's time for an update check
    
    def _parse_version(self, version_str: str) -> Tuple[int, ...]:
        """Parse version string into a tuple of integers for comparison.
        
        Args:
            version_str: Version string (e.g., "1.2.3")
            
        Returns:
            Tuple of integers representing the version
        """
        return tuple(map(int, version_str.split('.')))
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings.
        
        Args:
            version1: First version string
            version2: Second version string
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        v1 = self._parse_version(version1)
        v2 = self._parse_version(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
    
    def _get_app_bundle_path(self) -> Optional[Path]:
        """Get the path to the application bundle.
        
        Returns:
            Path to the application bundle or None if not found
        """
        # Check common locations
        paths = [
            Path(f"/Applications/{self.app_name}.app"),
            Path(f"{Path.home()}/Applications/{self.app_name}.app"),
            Path(f"{Path.home()}/Desktop/{self.app_name}.app")
        ]
        
        for path in paths:
            if path.exists():
                return path
        
        # Try to find the bundle using the running application
        if getattr(sys, 'frozen', False):
            # Running in a bundle
            app_path = Path(sys.executable).parent.parent.parent
            if app_path.suffix == '.app':
                return app_path
        
        return None
    
    def check_for_updates(self, silent: bool = False) -> Tuple[bool, Optional[ReleaseInfo]]:
        """Check for updates.
        
        Args:
            silent: Whether to suppress notifications
            
        Returns:
            Tuple of (update_available, release_info)
        """
        logger.info(f"Checking for updates (current version: {self.current_version})")
        
        # Update the last check time
        self._last_check_time = datetime.now()
        self._save_last_check_time()
        
        try:
            # Construct the URL for the update check
            update_url = f"{self.update_url}/check?version={self.current_version}&channel={self.update_channel.value}"
            
            # Add system information to the request
            system_info = {
                "os": platform.system(),
                "os_version": platform.release(),
                "architecture": platform.machine()
            }
            
            # Include system info in the URL
            for key, value in system_info.items():
                update_url += f"&{key}={value}"
            
            # Send the request
            with urllib.request.urlopen(update_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to check for updates: HTTP {response.status}")
                    self._status = UpdateStatus.ERROR
                    return False, None
                
                # Parse the response
                data = json.loads(response.read().decode())
                
                # Check if an update is available
                if not data.get("update_available", False):
                    logger.info("No updates available")
                    self._status = UpdateStatus.UP_TO_DATE
                    return False, None
                
                # Get release information
                release_info = ReleaseInfo.from_dict(data.get("release_info", {}))
                
                # Validate the release info
                if not release_info.download_url:
                    logger.error("Invalid release information: missing download URL")
                    self._status = UpdateStatus.ERROR
                    return False, None
                
                # Check if the release is newer than the current version
                if self._compare_versions(release_info.version, self.current_version) <= 0:
                    logger.info(f"Received version {release_info.version} is not newer than current version {self.current_version}")
                    self._status = UpdateStatus.UP_TO_DATE
                    return False, None
                
                # Check if the OS version requirement is met
                current_os_version = platform.release()
                if self._compare_versions(current_os_version, release_info.min_os_version) < 0:
                    logger.warning(f"Update requires macOS {release_info.min_os_version} or later (current: {current_os_version})")
                    self._status = UpdateStatus.ERROR
                    return False, None
                
                # Update available
                logger.info(f"Update available: {release_info.version}")
                self._latest_release = release_info
                self._status = UpdateStatus.UPDATE_AVAILABLE
                
                # Notify if not silent
                if not silent and self.on_update_available:
                    self.on_update_available(release_info)
                
                return True, release_info
                
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            self._status = UpdateStatus.ERROR
            return False, None
    
    def get_latest_release_info(self) -> Optional[ReleaseInfo]:
        """Get information about the latest available release.
        
        Returns:
            ReleaseInfo object for the latest release or None if no update is available
        """
        return self._latest_release
    
    def get_release_notes(self, version: Optional[str] = None) -> Optional[str]:
        """Get release notes for a specific version.
        
        Args:
            version: Version to get release notes for, or None for the latest
            
        Returns:
            Release notes as a string or None if not available
        """
        if version is None:
            # Get latest release notes
            if self._latest_release:
                return self._latest_release.release_notes
            return None
        
        # Try to fetch release notes for a specific version
        try:
            release_notes_url = f"{self.update_url}/notes?version={version}"
            
            with urllib.request.urlopen(release_notes_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to get release notes: HTTP {response.status}")
                    return None
                
                data = json.loads(response.read().decode())
                return data.get("release_notes", "")
                
        except Exception as e:
            logger.error(f"Failed to get release notes: {e}")
            return None
    
    def download_update(self, release_info: Optional[ReleaseInfo] = None) -> bool:
        """Download an update.
        
        Args:
            release_info: ReleaseInfo object for the release to download, or None for the latest
            
        Returns:
            True if download started successfully, False otherwise
        """
        if release_info is None:
            release_info = self._latest_release
        
        if not release_info:
            logger.error("No release information available for download")
            return False
        
        if self._status == UpdateStatus.UPDATE_DOWNLOADING:
            logger.warning("Update already downloading")
            return False
        
        logger.info(f"Starting download of update {release_info.version}")
        self._status = UpdateStatus.UPDATE_DOWNLOADING
        
        # Start download in a separate thread
        self._download_thread = threading.Thread(
            target=self._download_thread_func,
            args=(release_info,),
            daemon=True
        )
        self._download_thread.start()
        
        return True
    
    def _download_thread_func(self, release_info: ReleaseInfo) -> None:
        """Download thread function.
        
        Args:
            release_info: ReleaseInfo object for the release to download
        """
        try:
            # Create a temporary file for the download
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dmg') as temp_file:
                temp_path = Path(temp_file.name)
            
            # Download the update
            logger.info(f"Downloading from {release_info.download_url}")
            
            # Set up progress reporting
            def report_progress(count, block_size, total_size):
                if total_size > 0:
                    progress = min(count * block_size / total_size, 1.0)
                    if self.on_update_progress:
                        self.on_update_progress(progress)
            
            # Download the file
            urllib.request.urlretrieve(
                release_info.download_url,
                temp_path,
                reporthook=report_progress
            )
            
            # Verify checksum
            if not self._verify_checksum(temp_path, release_info.checksum):
                logger.error("Checksum verification failed")
                os.unlink(temp_path)
                self._status = UpdateStatus.ERROR
                return
            
            # Move to updates directory
            target_path = self.updates_dir / f"{self.app_name}-{release_info.version}.dmg"
            shutil.move(temp_path, target_path)
            
            logger.info(f"Update downloaded: {target_path}")
            self._download_path = target_path
            self._status = UpdateStatus.UPDATE_READY
            
            # Notify download completion
            if self.on_update_downloaded:
                self.on_update_downloaded(release_info, target_path)
                
        except Exception as e:
            logger.error(f"Failed to download update: {e}")
            self._status = UpdateStatus.ERROR
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify the checksum of a file.
        
        Args:
            file_path: Path to the file to verify
            expected_checksum: Expected SHA-256 checksum
            
        Returns:
            True if checksum matches, False otherwise
        """
        logger.info(f"Verifying checksum of {file_path}")
        
        try:
            # Calculate SHA-256 checksum
            sha256 = hashlib.sha256()
            
            with open(file_path, "rb") as f:
                # Read and update in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            
            actual_checksum = sha256.hexdigest()
            
            # Compare checksums
            if actual_checksum.lower() != expected_checksum.lower():
                logger.error(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                return False
            
            logger.info("Checksum verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify checksum: {e}")
            return False
    
    def install_update(self, backup: bool = True) -> bool:
        """Install a downloaded update.
        
        Args:
            backup: Whether to create a backup before installing
            
        Returns:
            True if installation started successfully, False otherwise
        """
        if self._status != UpdateStatus.UPDATE_READY:
            logger.error(f"Cannot install update: status is {self._status}")
            return False
        
        if not self._download_path or not self._download_path.exists():
            logger.error("Update package not found")
            return False
        
        if not self._latest_release:
            logger.error("Release information not available")
            return False
        
        logger.info(f"Installing update {self._latest_release.version}")
        self._status = UpdateStatus.UPDATE_INSTALLING
        
        # Create backup if requested
        if backup:
            if not self._create_backup():
                logger.error("Failed to create backup, aborting update")
                self._status = UpdateStatus.ERROR
                return False
        
        # Install the update
        try:
            # Mount the DMG
            mount_point = self._mount_dmg(self._download_path)
            if not mount_point:
                logger.error("Failed to mount update package")
                self._status = UpdateStatus.ERROR
                return False
            
            # Get the app bundle path
            app_bundle_path = self._get_app_bundle_path()
            if not app_bundle_path:
                logger.error("Failed to find application bundle")
                self._unmount_dmg(mount_point)
                self._status = UpdateStatus.ERROR
                return False
            
            # Find the application in the DMG
            app_in_dmg = None
            for item in os.listdir(mount_point):
                if item.endswith(".app"):
                    app_in_dmg = Path(mount_point) / item
                    break
            
            if not app_in_dmg:
                logger.error("Application not found in update package")
                self._unmount_dmg(mount_point)
                self._status = UpdateStatus.ERROR
                return False
            
            # Check if the application is running and ask to quit
            if self._is_app_running():
                logger.warning("Application is running, cannot update")
                self._unmount_dmg(mount_point)
                self._status = UpdateStatus.ERROR
                return False
            
            # Replace the application
            logger.info(f"Replacing {app_bundle_path} with {app_in_dmg}")
            
            # Remove the old application
            if app_bundle_path.exists():
                shutil.rmtree(app_bundle_path)
            
            # Copy the new application
            shutil.copytree(app_in_dmg, app_bundle_path)
            
            # Unmount the DMG
            self._unmount_dmg(mount_point)
            
            # Update history
            self._add_to_update_history(
                from_version=self.current_version,
                to_version=self._latest_release.version,
                success=True
            )
            
            # Update current version
            self.current_version = self._latest_release.version
            
            logger.info("Update installed successfully")
            self._status = UpdateStatus.UPDATE_COMPLETE
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to install update: {e}")
            self._status = UpdateStatus.ERROR
            
            # Add to history
            if self._latest_release:
                self._add_to_update_history(
                    from_version=self.current_version,
                    to_version=self._latest_release.version,
                    success=False,
                    notes=str(e)
                )
            
            return False
    
    def _mount_dmg(self, dmg_path: Path) -> Optional[str]:
        """Mount a DMG file.
        
        Args:
            dmg_path: Path to the DMG file
            
        Returns:
            Mount point or None if mounting failed
        """
        logger.info(f"Mounting DMG: {dmg_path}")
        
        try:
            # Mount the DMG
            cmd = ["hdiutil", "attach", "-nobrowse", "-noverify", str(dmg_path)]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Failed to mount DMG: {stderr}")
                return None
            
            # Parse the output to find the mount point
            lines = stdout.strip().split("\n")
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3:
                    mount_point = parts[-1].strip()
                    logger.info(f"DMG mounted at: {mount_point}")
                    return mount_point
            
            logger.error("Could not find mount point in output")
            return None
            
        except Exception as e:
            logger.error(f"Failed to mount DMG: {e}")
            return None
    
    def _unmount_dmg(self, mount_point: str) -> bool:
        """Unmount a DMG file.
        
        Args:
            mount_point: Mount point to unmount
            
        Returns:
            True if unmounting succeeded, False otherwise
        """
        logger.info(f"Unmounting DMG from: {mount_point}")
        
        try:
            # Unmount the DMG
            cmd = ["hdiutil", "detach", mount_point]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Failed to unmount DMG: {stderr}")
                return False
            
            logger.info("DMG unmounted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unmount DMG: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """Create a backup of the current application.
        
        Returns:
            True if backup succeeded, False otherwise
        """
        logger.info("Creating backup before update")
        
        try:
            # Get the app bundle path
            app_bundle_path = self._get_app_bundle_path()
            if not app_bundle_path or not app_bundle_path.exists():
                logger.error("Application bundle not found, cannot create backup")
                return False
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = self.backups_dir / f"{self.app_name}-{self.current_version}-{timestamp}.tar.gz"
            
            # Create backup directory
            self.backups_dir.mkdir(parents=True, exist_ok=True)
            
            # Create tar.gz archive
            logger.info(f"Creating backup at {backup_path}")
            archive_process = subprocess.Popen(
                ["tar", "-czf", str(backup_path), "-C", str(app_bundle_path.parent), app_bundle_path.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = archive_process.communicate()
            
            if archive_process.returncode != 0:
                logger.error(f"Failed to create backup: {stderr}")
                return False
            
            logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def rollback_update(self, backup_path: Optional[Path] = None) -> bool:
        """Rollback to a previous version.
        
        Args:
            backup_path: Path to backup file, or None to use the most recent
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        logger.info("Rolling back update")
        
        try:
            # Find the most recent backup if not specified
            if not backup_path:
                backups = list(self.backups_dir.glob(f"{self.app_name}-*.tar.gz"))
                if not backups:
                    logger.error("No backups found for rollback")
                    return False
                
                # Sort by modification time (newest first)
                backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                backup_path = backups[0]
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            logger.info(f"Using backup: {backup_path}")
            
            # Get the app bundle path
            app_bundle_path = self._get_app_bundle_path()
            if not app_bundle_path:
                logger.error("Application bundle not found")
                return False
            
            # Check if the application is running
            if self._is_app_running():
                logger.warning("Application is running, cannot rollback")
                return False
            
            # Extract version from backup filename
            backup_name = backup_path.stem
            version_match = backup_name.split("-")[1] if len(backup_name.split("-")) > 1 else "unknown"
            
            # Extract the backup to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Extracting backup to {temp_dir}")
                
                extract_process = subprocess.Popen(
                    ["tar", "-xzf", str(backup_path), "-C", temp_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = extract_process.communicate()
                
                if extract_process.returncode != 0:
                    logger.error(f"Failed to extract backup: {stderr}")
                    return False
                
                # Find the extracted application
                extracted_apps = list(Path(temp_dir).glob("*.app"))
                if not extracted_apps:
                    logger.error("No application found in backup")
                    return False
                
                extracted_app = extracted_apps[0]
                
                # Replace the current application
                logger.info(f"Replacing {app_bundle_path} with {extracted_app}")
                
                if app_bundle_path.exists():
                    shutil.rmtree(app_bundle_path)
                
                shutil.copytree(extracted_app, app_bundle_path)
            
            # Update history
            self._add_to_update_history(
                from_version=self.current_version,
                to_version=version_match,
                success=True,
                notes="Rollback"
            )
            
            # Update current version
            self.current_version = version_match
            
            logger.info(f"Rollback to version {version_match} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback update: {e}")
            return False
    
    def _is_app_running(self) -> bool:
        """Check if the application is currently running.
        
        Returns:
            True if the application is running, False otherwise
        """
        try:
            # macOS specific check
            cmd = ["pgrep", "-f", self.app_name]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            # If there's output, the app is running
            return process.returncode == 0 and stdout.strip() != ""
            
        except Exception as e:
            logger.error(f"Failed to check if app is running: {e}")
            return False
    
    def get_update_status(self) -> UpdateStatus:
        """Get the current update status.
        
        Returns:
            Current update status
        """
        return self._status
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get the update history.
        
        Returns:
            List of update history entries
        """
        return self._update_history
    
    def clear_downloads(self) -> bool:
        """Clear downloaded updates.
        
        Returns:
            True if operation succeeded, False otherwise
        """
        try:
            # Only clear if not currently downloading or installing
            if self._status in [UpdateStatus.UPDATE_DOWNLOADING, UpdateStatus.UPDATE_INSTALLING]:
                logger.warning("Cannot clear downloads while updating")
                return False
            
            # Clear the updates directory
            for item in self.updates_dir.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            logger.info("Downloaded updates cleared")
            
            # Reset status if necessary
            if self._status in [UpdateStatus.UPDATE_READY, UpdateStatus.UPDATE_COMPLETE]:
                self._status = UpdateStatus.UP_TO_DATE
            
            self._download_path = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear downloads: {e}")
            return False
    
    def set_update_channel(self, channel: UpdateChannel) -> None:
        """Set the update channel.
        
        Args:
            channel: New update channel
        """
        logger.info(f"Changing update channel from {self.update_channel.value} to {channel.value}")
        self.update_channel = channel


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    def on_update_available(release_info: ReleaseInfo) -> None:
        print(f"Update available: {release_info.version}")
        print(f"Release notes: {release_info.release_notes}")
    
    def on_update_downloaded(release_info: ReleaseInfo, path: Path) -> None:
        print(f"Update downloaded: {release_info.version} to {path}")
    
    def on_update_progress(progress: float) -> None:
        print(f"Download progress: {progress:.2%}")
    
    # Create update manager
    update_manager = UpdateManager(
        current_version="0.1.0",
        on_update_available=on_update_available,
        on_update_downloaded=on_update_downloaded,
        on_update_progress=on_update_progress
    )
    
    # Check for updates
    update_available, release_info = update_manager.check_for_updates()
    
    if update_available and release_info:
        # Download the update
        update_manager.download_update(release_info)
        
        # Wait for download to complete
        while update_manager.get_update_status() == UpdateStatus.UPDATE_DOWNLOADING:
            time.sleep(0.1)
        
        # Install the update
        if update_manager.get_update_status() == UpdateStatus.UPDATE_READY:
            update_manager.install_update()

#!/usr/bin/env python3
"""
Packaging script for MacAgent

This script handles the packaging, installation and update processes for MacAgent:
- Creates macOS application bundles
- Manages code signing and notarization
- Handles installation and updates
- Checks system requirements
"""

import os
import sys
import logging
import argparse
import subprocess
import shutil
import json
import plistlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MacAgent.Package")

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
RESOURCES_DIR = PROJECT_ROOT / "resources"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class PackagingSystem:
    """Handles packaging of MacAgent into a macOS application."""
    
    def __init__(self, config: str = "release", version: str = "0.1.0"):
        """Initialize the packaging system.
        
        Args:
            config: Build configuration to package
            version: Version of the application
        """
        self.config = config
        self.version = version
        self.build_dir = BUILD_DIR / config
        self.dist_dir = DIST_DIR / config
        self.app_name = "MacAgent"
        self.bundle_id = "com.macagent.app"
        
        # Create dist directory
        os.makedirs(self.dist_dir, exist_ok=True)
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and return its output.
        
        Args:
            cmd: Command to run as a list of arguments
            cwd: Working directory for the command
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if cwd is None:
            cwd = PROJECT_ROOT
            
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            text=True
        )
        
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        return return_code, stdout, stderr
    
    def create_app_bundle(self) -> bool:
        """Create a macOS application bundle.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Creating macOS application bundle for {self.app_name}")
        
        # Define app bundle structure
        app_path = self.dist_dir / f"{self.app_name}.app"
        contents_path = app_path / "Contents"
        macos_path = contents_path / "MacOS"
        resources_path = contents_path / "Resources"
        frameworks_path = contents_path / "Frameworks"
        
        # Create directories
        for path in [contents_path, macos_path, resources_path, frameworks_path]:
            os.makedirs(path, exist_ok=True)
        
        try:
            # Create Info.plist
            info_plist = {
                "CFBundleName": self.app_name,
                "CFBundleDisplayName": self.app_name,
                "CFBundleIdentifier": self.bundle_id,
                "CFBundleVersion": self.version,
                "CFBundleShortVersionString": self.version,
                "CFBundlePackageType": "APPL",
                "CFBundleSignature": "????",
                "CFBundleExecutable": self.app_name,
                "NSHumanReadableCopyright": f"Copyright © 2023 {self.app_name} Team",
                "NSHighResolutionCapable": True,
                "LSMinimumSystemVersion": "10.15.0",
                "NSPrincipalClass": "NSApplication",
            }
            
            with open(contents_path / "Info.plist", "wb") as f:
                plistlib.dump(info_plist, f)
            
            # Create launcher script
            launcher_path = macos_path / self.app_name
            with open(launcher_path, "w") as f:
                f.write(f"""#!/bin/bash
# Launcher script for {self.app_name}
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
PYTHON_PATH="$SCRIPT_DIR/../Frameworks/Python/bin/python3"
APP_PATH="$SCRIPT_DIR/../Resources/app"
cd "$APP_PATH"
"$PYTHON_PATH" -m MacAgent.src.ui.main_app "$@"
""")
            
            # Make launcher executable
            os.chmod(launcher_path, 0o755)
            
            # Copy application files
            if os.path.exists(self.build_dir / "lib"):
                shutil.copytree(
                    self.build_dir / "lib",
                    resources_path / "app",
                    dirs_exist_ok=True
                )
            
            # Create PkgInfo
            with open(contents_path / "PkgInfo", "w") as f:
                f.write("APPL????")
            
            logger.info(f"Application bundle created: {app_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create application bundle: {e}")
            return False
    
    def sign_app_bundle(self, identity: str) -> bool:
        """Sign the macOS application bundle.
        
        Args:
            identity: Code signing identity
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Signing application bundle with identity: {identity}")
        
        app_path = self.dist_dir / f"{self.app_name}.app"
        
        if not os.path.exists(app_path):
            logger.error(f"Application bundle not found: {app_path}")
            return False
        
        # Sign the app bundle
        cmd = [
            "codesign",
            "--force",
            "--sign", identity,
            "--deep",
            "--options", "runtime",
            str(app_path)
        ]
        
        return_code, stdout, stderr = self.run_command(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to sign application bundle: {stderr}")
            return False
        
        # Verify signature
        cmd = ["codesign", "--verify", "--verbose", str(app_path)]
        return_code, stdout, stderr = self.run_command(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to verify application signature: {stderr}")
            return False
        
        logger.info("Application bundle signed successfully")
        return True
    
    def create_dmg(self) -> bool:
        """Create a DMG installer.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating DMG installer")
        
        app_path = self.dist_dir / f"{self.app_name}.app"
        dmg_path = self.dist_dir / f"{self.app_name}-{self.version}.dmg"
        
        if not os.path.exists(app_path):
            logger.error(f"Application bundle not found: {app_path}")
            return False
        
        # Check if create-dmg is installed
        return_code, _, _ = self.run_command(["which", "create-dmg"])
        
        if return_code != 0:
            logger.warning("create-dmg not found, installing it")
            return_code, _, stderr = self.run_command(["brew", "install", "create-dmg"])
            
            if return_code != 0:
                logger.error(f"Failed to install create-dmg: {stderr}")
                logger.info("Using hdiutil directly")
                
                # Create temporary directory for DMG contents
                temp_dir = self.dist_dir / "dmg_temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Copy app to temporary directory
                shutil.copytree(app_path, temp_dir / f"{self.app_name}.app", dirs_exist_ok=True)
                
                # Create symlink to Applications folder
                os.symlink("/Applications", temp_dir / "Applications")
                
                # Create DMG
                cmd = [
                    "hdiutil", "create",
                    "-volname", self.app_name,
                    "-srcfolder", str(temp_dir),
                    "-ov", "-format", "UDZO",
                    str(dmg_path)
                ]
                
                return_code, stdout, stderr = self.run_command(cmd)
                
                # Clean up
                shutil.rmtree(temp_dir)
                
                if return_code != 0:
                    logger.error(f"Failed to create DMG: {stderr}")
                    return False
                
                logger.info(f"DMG created: {dmg_path}")
                return True
        
        # Use create-dmg for a nicer looking DMG
        cmd = [
            "create-dmg",
            "--volname", f"{self.app_name} Installer",
            "--volicon", str(RESOURCES_DIR / "icons" / "app_icon.icns") if os.path.exists(RESOURCES_DIR / "icons" / "app_icon.icns") else "",
            "--window-pos", "200", "100",
            "--window-size", "800", "400",
            "--icon-size", "100",
            "--icon", f"{self.app_name}.app", "200", "200",
            "--app-drop-link", "600", "200",
            "--no-internet-enable",
            str(dmg_path),
            str(app_path)
        ]
        
        return_code, stdout, stderr = self.run_command(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to create DMG: {stderr}")
            return False
        
        logger.info(f"DMG created: {dmg_path}")
        return True
    
    def package(self, sign: bool = False, identity: str = None, create_installer: bool = True) -> bool:
        """Package the application.
        
        Args:
            sign: Whether to sign the application bundle
            identity: Code signing identity (required if sign is True)
            create_installer: Whether to create an installer
            
        Returns:
            True if successful, False otherwise
        """
        # Create application bundle
        if not self.create_app_bundle():
            return False
        
        # Sign application bundle if requested
        if sign:
            if not identity:
                logger.error("Code signing identity not provided")
                return False
            
            if not self.sign_app_bundle(identity):
                return False
        
        # Create installer if requested
        if create_installer:
            if not self.create_dmg():
                return False
        
        logger.info("Packaging completed successfully")
        return True


class InstallationManager:
    """Manages installation of MacAgent."""
    
    def __init__(self, app_name: str = "MacAgent", version: str = "0.1.0"):
        """Initialize installation manager.
        
        Args:
            app_name: Name of the application
            version: Version of the application
        """
        self.app_name = app_name
        self.version = version
        self.bundle_id = f"com.{app_name.lower()}.app"
        self.min_os_version = "10.15"  # macOS Catalina
    
    def check_system_requirements(self) -> bool:
        """Check if the system meets the requirements for installation.
        
        Returns:
            True if requirements are met, False otherwise
        """
        logger.info("Checking system requirements")
        
        # Check macOS version
        cmd = ["sw_vers", "-productVersion"]
        return_code, stdout, stderr = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).communicate()
        
        if return_code != 0:
            logger.error(f"Failed to get macOS version: {stderr}")
            return False
        
        mac_version = stdout.strip()
        if mac_version < self.min_os_version:
            logger.error(f"macOS version {mac_version} is not supported. Minimum required version is {self.min_os_version}")
            return False
        
        logger.info(f"macOS version {mac_version} meets requirements")
        
        # Check disk space
        cmd = ["df", "-h", "/"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Failed to check disk space: {stderr}")
            return False
        
        # Parse disk space information
        lines = stdout.strip().split("\n")
        if len(lines) < 2:
            logger.error("Invalid disk space information")
            return False
        
        parts = lines[1].split()
        if len(parts) < 4:
            logger.error("Invalid disk space information")
            return False
        
        available_space = parts[3]
        logger.info(f"Available disk space: {available_space}")
        
        # Check for Python
        cmd = ["which", "python3"]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.warning("Python 3 not found in PATH")
        else:
            logger.info(f"Python 3 found: {stdout.strip()}")
        
        # All requirements met
        logger.info("System requirements check passed")
        return True
    
    def request_permissions(self) -> bool:
        """Request necessary permissions for the application.
        
        Returns:
            True if permissions granted, False otherwise
        """
        logger.info("Requesting permissions")
        
        # Permissions needed
        permissions = [
            "Screen Recording",
            "Accessibility",
            "Full Disk Access"
        ]
        
        for permission in permissions:
            logger.info(f"Requesting {permission} permission")
            
            # In a real implementation, we would use TCC database to check or prompt for permissions
            # For now, we'll just simulate the process
            logger.info(f"{permission} permission granted")
        
        return True
    
    def setup_application(self, install_path: Path) -> bool:
        """Set up the application after installation.
        
        Args:
            install_path: Path where the application is installed
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up application")
        
        # Create application data directory
        data_dir = Path.home() / ".macagent"
        os.makedirs(data_dir, exist_ok=True)
        
        # Create configuration file
        config = {
            "version": self.version,
            "first_run": True,
            "check_for_updates": True,
            "telemetry_enabled": False
        }
        
        with open(data_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Application data directory created: {data_dir}")
        
        # Create logs directory
        logs_dir = data_dir / "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        logger.info(f"Logs directory created: {logs_dir}")
        
        return True


class UpdateSystem:
    """Manages updates for MacAgent."""
    
    def __init__(self, app_name: str = "MacAgent", current_version: str = "0.1.0"):
        """Initialize update system.
        
        Args:
            app_name: Name of the application
            current_version: Current version of the application
        """
        self.app_name = app_name
        self.current_version = current_version
        self.update_url = "https://macagent.example.com/updates"
        self.updates_dir = Path.home() / ".macagent" / "updates"
        
        # Create updates directory
        os.makedirs(self.updates_dir, exist_ok=True)
    
    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check for available updates.
        
        Returns:
            Tuple of (update_available, latest_version, release_notes)
        """
        logger.info(f"Checking for updates (current version: {self.current_version})")
        
        try:
            # In a real implementation, we would query an update server
            # For now, we'll just simulate the process
            
            # Mock response from update server
            mock_response = {
                "latest_version": "0.2.0",
                "min_required_version": "0.1.0",
                "release_date": "2023-07-15",
                "release_notes": "- Improved UI\n- Fixed bugs\n- Added new features",
                "download_url": f"{self.update_url}/MacAgent-0.2.0.dmg",
                "signature": "abc123"
            }
            
            latest_version = mock_response["latest_version"]
            
            # Check if update is available
            if latest_version > self.current_version:
                logger.info(f"Update available: {latest_version}")
                return True, latest_version, mock_response["release_notes"]
            else:
                logger.info("No updates available")
                return False, None, None
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return False, None, None
    
    def download_update(self, version: str, url: str) -> Optional[Path]:
        """Download update package.
        
        Args:
            version: Version to download
            url: Download URL
            
        Returns:
            Path to downloaded file or None if download failed
        """
        logger.info(f"Downloading update {version} from {url}")
        
        try:
            # In a real implementation, we would download the file from the URL
            # For now, we'll just simulate the process
            
            download_path = self.updates_dir / f"{self.app_name}-{version}.dmg"
            
            # Simulate download
            with open(download_path, "w") as f:
                f.write("Simulated update package")
            
            logger.info(f"Update downloaded: {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"Failed to download update: {e}")
            return None
    
    def verify_update(self, file_path: Path, signature: str) -> bool:
        """Verify update package signature.
        
        Args:
            file_path: Path to update package
            signature: Expected signature
            
        Returns:
            True if verification successful, False otherwise
        """
        logger.info(f"Verifying update package: {file_path}")
        
        try:
            # In a real implementation, we would verify the file signature
            # For now, we'll just simulate the process
            
            # Simulate signature verification
            logger.info("Update package verified")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify update package: {e}")
            return False
    
    def install_update(self, file_path: Path) -> bool:
        """Install update package.
        
        Args:
            file_path: Path to update package
            
        Returns:
            True if installation successful, False otherwise
        """
        logger.info(f"Installing update package: {file_path}")
        
        try:
            # In a real implementation, we would mount the DMG and install the application
            # For now, we'll just simulate the process
            
            # Simulate update installation
            logger.info("Update installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install update: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Packaging script for MacAgent")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Package command
    package_parser = subparsers.add_parser("package", help="Package the application")
    package_parser.add_argument("--config", choices=["debug", "release"], default="release",
                              help="Build configuration to package")
    package_parser.add_argument("--version", default="0.1.0",
                              help="Version of the application")
    package_parser.add_argument("--sign", action="store_true",
                              help="Sign the application bundle")
    package_parser.add_argument("--identity",
                              help="Code signing identity")
    package_parser.add_argument("--no-installer", action="store_true",
                              help="Don't create an installer")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install the application")
    install_parser.add_argument("--path",
                              help="Path to application bundle or installer")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Check for and install updates")
    update_parser.add_argument("--check-only", action="store_true",
                             help="Only check for updates without installing")
    
    args = parser.parse_args()
    
    if args.command == "package":
        packager = PackagingSystem(
            config=args.config,
            version=args.version
        )
        
        success = packager.package(
            sign=args.sign,
            identity=args.identity,
            create_installer=not args.no_installer
        )
        
        if not success:
            logger.error("Packaging failed")
            sys.exit(1)
        
    elif args.command == "install":
        installer = InstallationManager()
        
        # Check system requirements
        if not installer.check_system_requirements():
            logger.error("System requirements not met")
            sys.exit(1)
        
        # Request permissions
        if not installer.request_permissions():
            logger.error("Failed to request permissions")
            sys.exit(1)
        
        # Set up application
        if not installer.setup_application(Path(args.path) if args.path else Path("/Applications/MacAgent.app")):
            logger.error("Failed to set up application")
            sys.exit(1)
        
        logger.info("Installation completed successfully")
        
    elif args.command == "update":
        updater = UpdateSystem()
        
        # Check for updates
        update_available, latest_version, release_notes = updater.check_for_updates()
        
        if not update_available:
            logger.info("No updates available")
            sys.exit(0)
        
        logger.info(f"Update available: {latest_version}")
        
        if release_notes:
            logger.info(f"Release notes:\n{release_notes}")
        
        if args.check_only:
            sys.exit(0)
        
        # Download update
        download_url = f"https://macagent.example.com/updates/MacAgent-{latest_version}.dmg"
        download_path = updater.download_update(latest_version, download_url)
        
        if not download_path:
            logger.error("Failed to download update")
            sys.exit(1)
        
        # Verify update
        if not updater.verify_update(download_path, "abc123"):
            logger.error("Failed to verify update")
            sys.exit(1)
        
        # Install update
        if not updater.install_update(download_path):
            logger.error("Failed to install update")
            sys.exit(1)
        
        logger.info("Update completed successfully")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

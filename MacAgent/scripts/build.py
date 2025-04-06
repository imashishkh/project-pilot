#!/usr/bin/env python3
"""
Build script for MacAgent

This script handles the build process for MacAgent, including:
- Managing build configurations (debug, release)
- Compiling necessary components
- Handling dependencies
- Preparing resources
- Setting up the build environment

Usage:
    python build.py [options]

Options:
    --config=CONFIG    Build configuration (debug, release) [default: debug]
    --clean            Clean build artifacts before building
    --verbose          Verbose output
    --test             Run tests after building
    --docs             Generate documentation
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MacAgent.Build")

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
SRC_DIR = PROJECT_ROOT / "MacAgent"
RESOURCES_DIR = PROJECT_ROOT / "resources"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
VERSION_FILE = SRC_DIR / "version.py"


class BuildEnvironment:
    """Manages the build environment for MacAgent."""
    
    def __init__(self, config: str, verbose: bool = False, clean: bool = False):
        """Initialize build environment.
        
        Args:
            config: Build configuration (debug, release)
            verbose: Whether to enable verbose output
            clean: Whether to clean build artifacts before building
        """
        self.config = config
        self.verbose = verbose
        self.clean = clean
        self.python_path = sys.executable
        self.platform = platform.system()
        self.build_dir = BUILD_DIR / config
        self.dist_dir = DIST_DIR / config
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Ensure we're on macOS
        if self.platform != "Darwin":
            logger.error(f"Unsupported platform: {self.platform}. This build script is for macOS only.")
            sys.exit(1)
        
        # Create build directories
        self._create_directories()
        
        # Clean build artifacts if requested
        if clean:
            self._clean()
    
    def _create_directories(self) -> None:
        """Create necessary build directories."""
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.dist_dir, exist_ok=True)
        
        logger.debug(f"Created build directory: {self.build_dir}")
        logger.debug(f"Created distribution directory: {self.dist_dir}")
    
    def _clean(self) -> None:
        """Clean build artifacts."""
        logger.info(f"Cleaning build artifacts in {self.build_dir}")
        
        if os.path.exists(self.build_dir):
            for item in os.listdir(self.build_dir):
                item_path = os.path.join(self.build_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
    
    def get_version(self) -> str:
        """Get the current version of MacAgent.
        
        Returns:
            Current version string
        """
        version_dict = {}
        try:
            with open(VERSION_FILE, "r") as f:
                version_content = f.read()
                
                # Parse version variables
                for line in version_content.splitlines():
                    if line.startswith("__version__"):
                        version_dict["version"] = line.split("=")[1].strip().strip("'").strip('"')
                    elif line.startswith("__build__"):
                        version_dict["build"] = line.split("=")[1].strip().strip("'").strip('"')
        except FileNotFoundError:
            logger.warning(f"Version file not found: {VERSION_FILE}")
            version_dict = {"version": "0.1.0", "build": "1"}
        
        return f"{version_dict.get('version', '0.1.0')}.{version_dict.get('build', '1')}"
    
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
        
        if self.verbose:
            if stdout:
                logger.debug(f"Command stdout: {stdout}")
            if stderr:
                logger.debug(f"Command stderr: {stderr}")
        
        return return_code, stdout, stderr


class BuildSystem:
    """Manages the build process for MacAgent."""
    
    def __init__(self, env: BuildEnvironment):
        """Initialize build system.
        
        Args:
            env: Build environment
        """
        self.env = env
        self.version = env.get_version()
        
        # Build configuration
        self.debug = env.config == "debug"
        
        # Paths
        self.build_temp = env.build_dir / "temp"
        self.build_lib = env.build_dir / "lib"
        
        # Create build subdirectories
        os.makedirs(self.build_temp, exist_ok=True)
        os.makedirs(self.build_lib, exist_ok=True)
    
    def _install_dependencies(self) -> bool:
        """Install Python dependencies.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Installing dependencies")
        
        # Use pip to install dependencies to the build directory
        cmd = [
            self.env.python_path, "-m", "pip", "install",
            "--target", str(self.build_lib),
            "--requirement", str(REQUIREMENTS_FILE)
        ]
        
        if self.debug:
            cmd.append("--verbose")
        
        return_code, stdout, stderr = self.env.run_command(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to install dependencies: {stderr}")
            return False
        
        logger.info("Dependencies installed successfully")
        return True
    
    def _copy_source(self) -> bool:
        """Copy source files to build directory.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Copying source files")
        
        try:
            # Copy the MacAgent package
            source_dest = self.build_lib / "MacAgent"
            
            # Ensure the destination directory exists
            os.makedirs(source_dest, exist_ok=True)
            
            # Copy the source files
            shutil.copytree(
                SRC_DIR,
                source_dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd")
            )
            
            logger.info("Source files copied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy source files: {e}")
            return False
    
    def _copy_resources(self) -> bool:
        """Copy resources to build directory.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Copying resources")
        
        try:
            # Copy the resources
            resources_dest = self.build_lib / "resources"
            
            # Ensure the destination directory exists
            os.makedirs(resources_dest, exist_ok=True)
            
            # Check if resources directory exists
            if not os.path.exists(RESOURCES_DIR):
                logger.warning(f"Resources directory not found: {RESOURCES_DIR}")
                return True
            
            # Copy the resource files
            shutil.copytree(
                RESOURCES_DIR,
                resources_dest,
                dirs_exist_ok=True
            )
            
            logger.info("Resources copied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy resources: {e}")
            return False
    
    def _generate_build_info(self) -> bool:
        """Generate build information file.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Generating build information")
        
        try:
            build_info = {
                "version": self.version,
                "config": self.env.config,
                "build_time": subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).decode().strip(),
                "python_version": platform.python_version(),
                "system": {
                    "os": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine()
                }
            }
            
            # Write build information to file
            build_info_path = self.build_lib / "MacAgent" / "build_info.json"
            with open(build_info_path, "w") as f:
                json.dump(build_info, f, indent=2)
            
            logger.info(f"Build information generated: {build_info_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate build information: {e}")
            return False
    
    def _run_tests(self) -> bool:
        """Run tests.
        
        Returns:
            True if tests pass, False otherwise
        """
        logger.info("Running tests")
        
        tests_dir = PROJECT_ROOT / "tests"
        if not os.path.exists(tests_dir):
            logger.warning("Tests directory not found, skipping tests")
            return True
        
        cmd = [
            self.env.python_path, "-m", "pytest", str(tests_dir),
            "-v" if self.env.verbose else ""
        ]
        
        return_code, stdout, stderr = self.env.run_command(cmd)
        
        if return_code != 0:
            logger.error(f"Tests failed: {stderr}")
            return False
        
        logger.info("Tests passed")
        return True
    
    def _generate_docs(self) -> bool:
        """Generate documentation.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Generating documentation")
        
        docs_dir = PROJECT_ROOT / "docs"
        
        # Check if docs directory exists
        if not os.path.exists(docs_dir):
            logger.warning("Documentation directory not found, creating it")
            os.makedirs(docs_dir, exist_ok=True)
        
        # Check if Sphinx is installed
        cmd = [self.env.python_path, "-m", "pip", "show", "sphinx"]
        return_code, _, _ = self.env.run_command(cmd)
        
        if return_code != 0:
            logger.warning("Sphinx not installed, installing it")
            cmd = [self.env.python_path, "-m", "pip", "install", "sphinx"]
            return_code, _, stderr = self.env.run_command(cmd)
            
            if return_code != 0:
                logger.error(f"Failed to install Sphinx: {stderr}")
                return False
        
        # Generate documentation
        cmd = [
            self.env.python_path, "-m", "sphinx.cmd.build",
            "-b", "html",
            str(PROJECT_ROOT),
            str(docs_dir / "build")
        ]
        
        return_code, stdout, stderr = self.env.run_command(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to generate documentation: {stderr}")
            return False
        
        logger.info("Documentation generated")
        return True
    
    def build(self, run_tests: bool = False, generate_docs: bool = False) -> bool:
        """Execute the build process.
        
        Args:
            run_tests: Whether to run tests after building
            generate_docs: Whether to generate documentation
            
        Returns:
            True if build successful, False otherwise
        """
        logger.info(f"Building MacAgent version {self.version} ({self.env.config} configuration)")
        
        # Install dependencies
        if not self._install_dependencies():
            return False
        
        # Copy source code
        if not self._copy_source():
            return False
        
        # Copy resources
        if not self._copy_resources():
            return False
        
        # Generate build information
        if not self._generate_build_info():
            return False
        
        # Run tests if requested
        if run_tests and not self._run_tests():
            return False
        
        # Generate documentation if requested
        if generate_docs and not self._generate_docs():
            return False
        
        logger.info(f"Build completed successfully: {self.build_lib}")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build script for MacAgent")
    parser.add_argument("--config", choices=["debug", "release"], default="debug",
                       help="Build configuration (debug, release)")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build artifacts before building")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--test", action="store_true",
                       help="Run tests after building")
    parser.add_argument("--docs", action="store_true",
                       help="Generate documentation")
    
    args = parser.parse_args()
    
    # Create build environment
    env = BuildEnvironment(
        config=args.config,
        verbose=args.verbose,
        clean=args.clean
    )
    
    # Create build system
    build_system = BuildSystem(env)
    
    # Execute build
    success = build_system.build(
        run_tests=args.test,
        generate_docs=args.docs
    )
    
    if not success:
        logger.error("Build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

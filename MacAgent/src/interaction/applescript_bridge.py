"""
AppleScript Bridge Module for MacAgent

This module provides a bridge between Python and AppleScript, allowing the execution
of AppleScript commands with proper error handling, timeout management, and result parsing.
"""

import os
import re
import time
import uuid
import logging
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from threading import Timer
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AppleScriptBridge:
    """
    A bridge for executing AppleScript commands from Python.
    
    This class handles script compilation, execution, error handling,
    and result parsing for AppleScript operations.
    """
    
    # AppleScript result type patterns
    _TYPE_PATTERNS = {
        'integer': r'^(-?\d+)$',
        'float': r'^(-?\d+\.\d+)$',
        'boolean': r'^(true|false)$',
        'list': r'^\{(.+)\}$',
        'record': r'^{(.+:.+)(, .+:.+)*}$',
        'date': r'^date "(.+)"$',
        'null': r'^missing value$'
    }
    
    # Common AppleScript syntax fixes
    _APPLESCRIPT_PREAMBLE = """
    use scripting additions
    use framework "Foundation"
    use framework "AppKit"
    """
    
    # Cache for compiled scripts
    _SCRIPT_CACHE = {}
    
    def __init__(self, 
                 cache_compiled_scripts: bool = True,
                 default_timeout: float = 30.0,
                 log_scripts: bool = True,
                 debug_mode: bool = False):
        """
        Initialize the AppleScript bridge.
        
        Args:
            cache_compiled_scripts: Whether to cache compiled scripts for reuse
            default_timeout: Default timeout for script execution in seconds
            log_scripts: Whether to log executed scripts (may contain sensitive info)
            debug_mode: Enable additional debug logging
        """
        self.cache_compiled_scripts = cache_compiled_scripts
        self.default_timeout = default_timeout
        self.log_scripts = log_scripts
        self.debug_mode = debug_mode
        
        # Script execution statistics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timed_out_executions': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Check if osascript is available
        self._check_osascript()
        
        logger.info(f"AppleScriptBridge initialized with timeout={default_timeout}s, "
                  f"cache={cache_compiled_scripts}, debug={debug_mode}")
    
    def _check_osascript(self) -> bool:
        """
        Check if osascript command is available.
        
        Returns:
            True if osascript is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['which', 'osascript'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.debug(f"osascript found at: {result.stdout.strip()}")
                return True
            else:
                logger.warning("osascript command not found. AppleScript functionality will be limited.")
                return False
                
        except Exception as e:
            logger.error(f"Error checking for osascript: {e}")
            return False
    
    def run_script(self, 
                  script: str, 
                  timeout: float = None, 
                  variables: Dict[str, Any] = None,
                  return_error_output: bool = False) -> Tuple[bool, Any]:
        """
        Execute an AppleScript script and return the result.
        
        Args:
            script: The AppleScript code to execute
            timeout: Timeout in seconds (None to use default)
            variables: Dictionary of variables to substitute in the script
            return_error_output: Whether to return error output if the script fails
            
        Returns:
            Tuple of (success, result/error)
        """
        # Update execution statistics
        self.execution_stats['total_executions'] += 1
        
        # Apply variable substitution if provided
        if variables:
            script = self._substitute_variables(script, variables)
        
        # Add preamble to ensure consistent behavior and avoid syntax issues
        script = self._APPLESCRIPT_PREAMBLE + "\n" + script
        
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout
            
        # Add proper timeout handling to the script if it's not a simple one-liner
        if "\n" in script and not "with timeout" in script:
            # Wrap script in timeout block for proper AppleScript timeout handling
            script = f"""
with timeout of {int(timeout)} seconds
    {script}
end timeout
"""
        
        # Log the script (if enabled)
        if self.log_scripts:
            # Truncate long scripts for logging
            log_script = script if len(script) < 500 else f"{script[:500]}... [truncated]"
            logger.debug(f"Executing AppleScript (timeout={timeout}s):\n{log_script}")
        
        start_time = time.time()
        timed_out = False
        
        try:
            # Create a process to run the script
            process = subprocess.Popen(
                ['osascript', '-e', script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                # Wait for the process with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                # Update timing statistics
                total_time = self.execution_stats['total_execution_time'] + execution_time
                self.execution_stats['total_execution_time'] = total_time
                
                num_success = self.execution_stats['successful_executions']
                self.execution_stats['average_execution_time'] = (
                    total_time / (num_success + 1) if num_success > 0 else total_time
                )
                
                # Check for errors
                if process.returncode != 0:
                    self.execution_stats['failed_executions'] += 1
                    error_message = stderr.strip()
                    logger.error(f"AppleScript execution failed: {error_message}")
                    return False, error_message if return_error_output else None
                
                # Parse the result
                result = self._parse_result(stdout.strip())
                
                # Log success
                self.execution_stats['successful_executions'] += 1
                logger.debug(f"AppleScript executed successfully in {execution_time:.3f}s")
                
                return True, result
            
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                timed_out = True
                self.execution_stats['timed_out_executions'] += 1
                
                execution_time = time.time() - start_time
                logger.warning(f"AppleScript execution timed out after {timeout} seconds")
                
                # Try to get any output from the terminated process
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                    if stderr:
                        logger.debug(f"Error output from timed-out script: {stderr}")
                except Exception:
                    pass  # Ignore errors from the terminated process
                
                return False, f"Execution timed out after {timeout} seconds"
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.execution_stats['failed_executions'] += 1
            logger.error(f"Error executing AppleScript: {e}")
            
            return False, str(e) if return_error_output else None
    
    def compile_script(self, script: str) -> Optional[str]:
        """
        Compile an AppleScript script to a temporary file for repeated execution.
        
        Args:
            script: The AppleScript code to compile
            
        Returns:
            Path to the compiled script, or None if compilation failed
        """
        try:
            # Generate a unique script ID
            script_hash = hash(script)
            script_id = f"ascript_{script_hash}_{uuid.uuid4().hex[:8]}"
            
            # Check if already in cache
            if self.cache_compiled_scripts and script_hash in self._SCRIPT_CACHE:
                cached_path = self._SCRIPT_CACHE[script_hash]
                if os.path.exists(cached_path):
                    logger.debug(f"Using cached compiled script: {cached_path}")
                    return cached_path
            
            # Create a temporary file for the source script
            with tempfile.NamedTemporaryFile(suffix='.applescript', mode='w', delete=False) as source_file:
                source_path = source_file.name
                source_file.write(script)
            
            # Determine the output path
            compiled_path = os.path.join(
                tempfile.gettempdir(), 
                f"{script_id}.scpt"
            )
            
            # Compile the script
            result = subprocess.run(
                ['osacompile', '-o', compiled_path, source_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            # Clean up the source file
            os.unlink(source_path)
            
            if result.returncode != 0:
                logger.error(f"Failed to compile script: {result.stderr.strip()}")
                return None
            
            # Cache the compiled script path
            if self.cache_compiled_scripts:
                self._SCRIPT_CACHE[script_hash] = compiled_path
            
            logger.debug(f"Script compiled successfully: {compiled_path}")
            return compiled_path
            
        except Exception as e:
            logger.error(f"Error compiling AppleScript: {e}")
            return None
    
    def run_compiled_script(self, 
                          script_path: str, 
                          timeout: float = None,
                          parameters: List[str] = None,
                          return_error_output: bool = False) -> Tuple[bool, Any]:
        """
        Execute a compiled AppleScript file.
        
        Args:
            script_path: Path to the compiled script
            timeout: Timeout in seconds (None to use default)
            parameters: List of parameters to pass to the script
            return_error_output: Whether to return error output if the script fails
            
        Returns:
            Tuple of (success, result/error)
        """
        # Check if file exists
        if not os.path.exists(script_path):
            logger.error(f"Compiled script not found: {script_path}")
            return False, f"Script not found: {script_path}"
        
        # Update execution statistics
        self.execution_stats['total_executions'] += 1
        
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout
        
        # Build the command
        cmd = ['osascript', script_path]
        
        # Add parameters if provided
        if parameters:
            cmd.extend(parameters)
        
        logger.debug(f"Executing compiled script: {script_path} (timeout={timeout}s)")
        
        start_time = time.time()
        
        try:
            # Create a process to run the script
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set up timeout
            kill_process = lambda p: p.kill()
            timer = Timer(timeout, kill_process, [process])
            
            try:
                timer.start()
                stdout, stderr = process.communicate()
                
                execution_time = time.time() - start_time
                
                # Update timing statistics
                total_time = self.execution_stats['total_execution_time'] + execution_time
                self.execution_stats['total_execution_time'] = total_time
                
                num_success = self.execution_stats['successful_executions']
                self.execution_stats['average_execution_time'] = (
                    total_time / (num_success + 1) if num_success > 0 else total_time
                )
                
                # Check if process timed out (returncode is None)
                if process.returncode is None:
                    self.execution_stats['timed_out_executions'] += 1
                    logger.warning(f"Compiled script execution timed out after {timeout} seconds")
                    return False, f"Execution timed out after {timeout} seconds"
                
                # Check for errors
                if process.returncode != 0:
                    self.execution_stats['failed_executions'] += 1
                    error_message = stderr.strip()
                    logger.error(f"Compiled script execution failed: {error_message}")
                    return False, error_message if return_error_output else None
                
                # Parse the result
                result = self._parse_result(stdout.strip())
                
                # Log success
                self.execution_stats['successful_executions'] += 1
                logger.debug(f"Compiled script executed successfully in {execution_time:.3f}s")
                
                return True, result
                
            finally:
                timer.cancel()
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.execution_stats['failed_executions'] += 1
            logger.error(f"Error executing compiled script: {e}")
            
            return False, str(e) if return_error_output else None
    
    def _parse_result(self, result_str: str) -> Any:
        """
        Parse AppleScript result string into appropriate Python data types.
        
        Args:
            result_str: The result string from AppleScript
            
        Returns:
            Parsed Python object
        """
        if not result_str:
            return None
        
        # Check for null value
        if re.match(self._TYPE_PATTERNS['null'], result_str):
            return None
        
        # Check for integer
        if re.match(self._TYPE_PATTERNS['integer'], result_str):
            return int(result_str)
        
        # Check for float
        if re.match(self._TYPE_PATTERNS['float'], result_str):
            return float(result_str)
        
        # Check for boolean
        if re.match(self._TYPE_PATTERNS['boolean'], result_str):
            return result_str.lower() == 'true'
        
        # Check for list
        list_match = re.match(self._TYPE_PATTERNS['list'], result_str)
        if list_match:
            # Parse list items (this is simplified and might not handle all cases)
            items_str = list_match.group(1)
            items = []
            
            # Simple splitting by comma (doesn't handle nested structures well)
            for item in items_str.split(','):
                items.append(self._parse_result(item.strip()))
            
            return items
        
        # Check for date
        date_match = re.match(self._TYPE_PATTERNS['date'], result_str)
        if date_match:
            date_str = date_match.group(1)
            try:
                # Try to parse the date
                return datetime.strptime(date_str, '%A, %B %d, %Y at %I:%M:%S %p')
            except ValueError:
                # If date parsing fails, return as is
                return date_str
        
        # Default to string (remove quotes if present)
        if result_str.startswith('"') and result_str.endswith('"'):
            return result_str[1:-1]
        
        return result_str
    
    def _substitute_variables(self, script: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in an AppleScript template.
        
        Args:
            script: The AppleScript template
            variables: Dictionary of variables to substitute
            
        Returns:
            Script with variables substituted
        """
        # Create a copy of the script
        result = script
        
        # Add variable initialization statements at the beginning for all variables
        var_init_lines = []
        for var_name, var_value in variables.items():
            # Skip if the variable is already initialized in the script
            if f"set {var_name} to " in script:
                continue
                
            # Add initialization
            as_value = self._python_to_applescript(var_value)
            var_init_lines.append(f"set {var_name} to {as_value}")
        
        # If there are initializations, add them after the first non-comment line
        if var_init_lines:
            # Find the first non-comment, non-empty line
            lines = result.split("\n")
            insert_pos = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("--") and not stripped.startswith("#"):
                    insert_pos = i + 1
                    break
            
            # Insert the variable initializations
            init_block = "\n".join(var_init_lines)
            lines.insert(insert_pos, "\n" + init_block + "\n")
            result = "\n".join(lines)
        
        # Substitute each variable
        for var_name, var_value in variables.items():
            # Create the placeholder pattern
            placeholder = f"${var_name}$"
            
            # Convert value to AppleScript compatible format
            as_value = self._python_to_applescript(var_value)
            
            # Replace all occurrences
            result = result.replace(placeholder, as_value)
        
        return result
    
    def _python_to_applescript(self, value: Any) -> str:
        """
        Convert Python value to AppleScript string representation.
        
        Args:
            value: Python value to convert
            
        Returns:
            AppleScript string representation
        """
        if value is None:
            return "missing value"
        
        if isinstance(value, bool):
            return "true" if value else "false"
        
        if isinstance(value, (int, float)):
            return str(value)
        
        if isinstance(value, list):
            # Ensure proper list comparison when nested lists are involved
            # If the list contains only one item that is also a list, use double brackets
            if len(value) == 1 and isinstance(value[0], list):
                inner_items = [self._python_to_applescript(item) for item in value[0]]
                return "{{" + ", ".join(inner_items) + "}}"
            else:
                items = [self._python_to_applescript(item) for item in value]
                return "{" + ", ".join(items) + "}"
        
        if isinstance(value, dict):
            items = [f"{k}:{self._python_to_applescript(v)}" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        
        if isinstance(value, datetime):
            date_str = value.strftime('%A, %B %d, %Y at %I:%M:%S %p')
            return f'date "{date_str}"'
        
        # For strings, properly escape for AppleScript
        if isinstance(value, str):
            # If the value already looks like an AppleScript literal (starts with ")
            # and was passed in that way intentionally, use it as is
            if value.startswith('"') and value.endswith('"'):
                return value
                
            # Otherwise, escape " as \" for AppleScript strings
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        
        # Default for any other type: convert to string and escape
        escaped = str(value).replace('"', '\\"')
        return f'"{escaped}"'
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about script executions.
        
        Returns:
            Dictionary of execution statistics
        """
        return self.execution_stats.copy()
    
    def clear_cache(self) -> None:
        """Clear the compiled script cache."""
        if self.cache_compiled_scripts:
            # Delete cached script files
            for script_path in self._SCRIPT_CACHE.values():
                try:
                    if os.path.exists(script_path):
                        os.unlink(script_path)
                except Exception as e:
                    logger.warning(f"Error deleting cached script {script_path}: {e}")
            
            # Clear the cache dictionary
            self._SCRIPT_CACHE.clear()
            
            logger.debug("Compiled script cache cleared")
    
    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.clear_cache()

    def _generate_wait_function(self, condition: str, max_time: float = 30.0, interval: float = 0.5) -> str:
        """
        Generate a proper idle-based waiting function for AppleScript.
        
        Args:
            condition: The condition to wait for (AppleScript boolean expression)
            max_time: Maximum time to wait in seconds
            interval: Check interval in seconds
            
        Returns:
            AppleScript code for efficient waiting
        """
        return f"""
-- Efficient waiting function that doesn't block the system
on waitFor(maxTime)
    set startTime to current date
    set endTime to startTime + maxTime
    
    repeat until ({condition}) or (current date > endTime)
        delay {interval} -- Allow other processes to run
    end repeat
    
    return ({condition})
end waitFor

set waitResult to waitFor({max_time})
return waitResult
"""

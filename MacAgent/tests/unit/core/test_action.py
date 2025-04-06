"""
Unit tests for the ActionModule parameter filtering functionality.

Tests the _filter_params function and related parameter handling functionality.
"""

import unittest
import asyncio
import inspect
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the functions to test
from MacAgent.src.core.action import _filter_params, _log_parameter_mismatch


class TestParameterFiltering(unittest.TestCase):
    """Test cases for parameter filtering functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create sample functions with different signatures for testing
        def simple_func(a, b, c=None):
            pass
        
        def func_with_required_params(x, y, z):
            pass
        
        def func_with_args_kwargs(a, b=2, *args, **kwargs):
            pass
        
        self.simple_func = simple_func
        self.func_with_required_params = func_with_required_params
        self.func_with_args_kwargs = func_with_args_kwargs
    
    def test_filter_params_matching_signature(self):
        """Test filtering parameters that match the function signature."""
        # Arrange
        params = {"a": 1, "b": 2, "c": 3}
        
        # Act
        result = asyncio.run(self._run_filter_params(self.simple_func, params))
        
        # Assert
        self.assertEqual(result, params)
        self.assertEqual(len(result), 3)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
    
    def test_filter_params_extra_params(self):
        """Test filtering parameters with extra parameters that don't match."""
        # Arrange
        params = {"a": 1, "b": 2, "c": 3, "d": 4, "extra": "value"}
        
        # Act
        result = asyncio.run(self._run_filter_params(self.simple_func, params))
        
        # Assert
        self.assertEqual(len(result), 3)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertNotIn("d", result)
        self.assertNotIn("extra", result)
    
    def test_filter_params_missing_params(self):
        """Test filtering parameters with missing parameters."""
        # Arrange
        params = {"a": 1}  # 'a' is not in the signature of func_with_required_params which expects x, y, z
        
        # Act
        result = asyncio.run(self._run_filter_params(self.func_with_required_params, params))
        
        # Assert
        self.assertEqual(len(result), 0)  # Changed from 1 to 0 since 'a' isn't in the signature
        self.assertNotIn("a", result)     # 'a' should be filtered out
        self.assertNotIn("b", result)
        self.assertNotIn("z", result)
    
    def test_filter_params_empty_dict(self):
        """Test filtering with an empty parameter dictionary."""
        # Arrange
        params = {}
        
        # Act
        result = asyncio.run(self._run_filter_params(self.simple_func, params))
        
        # Assert
        self.assertEqual(result, {})
        self.assertEqual(len(result), 0)
    
    def test_filter_params_args_kwargs(self):
        """Test filtering with functions that have *args and **kwargs."""
        # Arrange
        params = {"a": 1, "b": 2, "c": 3, "d": 4}
        
        # Act
        result = asyncio.run(self._run_filter_params(self.func_with_args_kwargs, params))
        
        # Assert
        self.assertEqual(len(result), 2)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertNotIn("c", result)  # Not in signature explicitly
        self.assertNotIn("d", result)  # Not in signature explicitly
    
    def test_filter_params_exception_handling(self):
        """Test exception handling in _filter_params."""
        # Arrange
        params = {"a": 1, "b": 2}
        
        # Mock inspect.signature to raise an exception
        with patch('inspect.signature', side_effect=ValueError("Test error")):
            # Act
            result = asyncio.run(_filter_params(self.simple_func, params))
            
            # Assert
            self.assertEqual(result, {})
    
    def test_log_parameter_mismatch_unexpected_params(self):
        """Test logging of unexpected parameters."""
        # Arrange
        func_name = "test_func"
        provided_params = {"a": 1, "b": 2, "extra": 3}
        sig = inspect.signature(self.simple_func)
        
        # Act & Assert - just verify it doesn't raise exceptions
        with patch('logging.Logger.warning') as mock_warning:
            asyncio.run(_log_parameter_mismatch(func_name, provided_params, sig.parameters))
            # Verify warning was called for unexpected parameters
            mock_warning.assert_any_call("Function 'test_func' received unexpected parameters: {'extra'}")
    
    def test_log_parameter_mismatch_missing_required(self):
        """Test logging of missing required parameters."""
        # Arrange
        func_name = "test_func"
        provided_params = {"x": 1}  # Missing 'y' and 'z'
        sig = inspect.signature(self.func_with_required_params)
        
        # Act & Assert
        with patch('logging.Logger.warning') as mock_warning:
            asyncio.run(_log_parameter_mismatch(func_name, provided_params, sig.parameters))
            # Verify warning was called for missing parameters
            # We don't check the exact parameters as ordering in sets can vary
            self.assertTrue(mock_warning.called)
    
    def test_log_parameter_mismatch_exception_handling(self):
        """Test exception handling in _log_parameter_mismatch."""
        # Arrange
        func_name = "test_func"
        provided_params = {"a": 1}
        
        # Create a mock that raises an exception when used
        bad_accepted_params = MagicMock()
        bad_accepted_params.keys.side_effect = Exception("Test error")
        
        # Act & Assert
        with patch('logging.Logger.error') as mock_error:
            asyncio.run(_log_parameter_mismatch(func_name, provided_params, bad_accepted_params))
            # Verify error was logged
            mock_error.assert_called_once()
    
    async def _run_filter_params(self, func, params):
        """Helper to run the async _filter_params function in tests."""
        # Mock _log_parameter_mismatch to prevent actual logging during tests
        with patch('MacAgent.src.core.action._log_parameter_mismatch', return_value=None):
            return await _filter_params(func, params)


if __name__ == '__main__':
    unittest.main()

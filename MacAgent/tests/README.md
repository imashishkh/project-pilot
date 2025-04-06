# MacAgent Testing Framework

This directory contains the testing framework for the MacAgent project. The tests are organized into a structured hierarchy to ensure maintainability and clarity.

## Directory Structure

```
tests/
├── conftest.py          # Common pytest fixtures and configuration
├── pytest.ini           # pytest configuration
├── README.md            # This file
├── unit/                # Unit tests for individual components
│   ├── vision/          # Vision system tests
│   ├── interaction/     # Interaction system tests
│   ├── intelligence/    # Intelligence system tests
│   └── core/            # Core system tests
├── integration/         # Integration tests between components
├── resources/           # Test resources (images, etc.)
│   └── images/          # Test images
└── output/              # Test output files
```

## Running Tests

To run all tests:

```bash
pytest
```

To run tests for a specific component:

```bash
pytest tests/unit/vision/
```

To run tests with a specific marker:

```bash
pytest -m vision
```

## Available Markers

- `vision`: Tests for the vision system
- `interaction`: Tests for the interaction system
- `intelligence`: Tests for the intelligence system
- `core`: Tests for the core system
- `integration`: Integration tests
- `slow`: Tests that take a long time to run

## Writing Tests

### Import Pattern

All tests should use absolute imports from the MacAgent package:

```python
from MacAgent.src.vision import ScreenCapture
```

The `conftest.py` file ensures that these imports will work correctly.

### Test File Naming

All test files should follow the naming convention:

```
test_<component_name>.py
```

### Test Function Naming

All test functions should follow the naming convention:

```python
def test_<functionality_being_tested>(...):
    ...
```

### Using Fixtures

Common fixtures are available in `conftest.py`:

```python
def test_something(test_output_dir):
    output_path = os.path.join(test_output_dir, "result.png")
    # Use output_path...
```

## Test Resources

Place any resources needed for tests in the `resources/` directory. Use the `test_resources_dir` fixture to access these resources:

```python
def test_with_resources(test_resources_dir):
    image_path = os.path.join(test_resources_dir, "images", "test_image.png")
    # Use image_path...
```

## Test Output

Tests should save any output files to the `output/` directory. Use the `test_output_dir` fixture to access this directory:

```python
def test_generating_output(test_output_dir):
    output_path = os.path.join(test_output_dir, "result.png")
    # Save output to output_path...
```

## Continuous Integration

The test framework is designed to work with CI systems. Tests that require special resources or take a long time should be marked with appropriate markers so they can be skipped in CI environments if needed. 
# MacAgent Developer Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Key Components](#key-components)
4. [Development Environment Setup](#development-environment-setup)
5. [Project Structure](#project-structure)
6. [Security Architecture](#security-architecture)
7. [Workflow System](#workflow-system)
8. [API Documentation](#api-documentation)
9. [Testing](#testing)
10. [Contributing](#contributing)
11. [Style Guide](#style-guide)
12. [Release Process](#release-process)

## Introduction

This guide is for developers who want to contribute to MacAgent or build extensions and integrations. MacAgent is an AI-powered assistant for macOS, designed with security, extensibility, and usability in mind.

### Development Philosophy

Our development approach emphasizes:

- **Security First**: Security is built into our architecture from the ground up
- **Modular Design**: Components are loosely coupled for extensibility
- **Comprehensive Testing**: All components have thorough test coverage
- **Clear Documentation**: Code and systems are well-documented
- **User Privacy**: User data never leaves their machine without explicit consent

## Architecture Overview

MacAgent follows a modular architecture with clear separation of concerns:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                    System Orchestrator                       │
└───┬───────────────┬───────────────┬────────────────┬────────┘
    │               │               │                │
┌───▼───┐       ┌───▼───┐       ┌───▼───┐        ┌───▼───┐
│Security│       │Workflow│       │App    │        │Utility│
│System  │       │System  │       │Systems│        │Systems│
└───┬───┘       └───┬───┘       └───┬───┘        └───┬───┘
    │               │               │                │
┌───▼───────────────▼───────────────▼────────────────▼───┐
│                  macOS Integration Layer                │
└─────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Component Isolation**: Each component operates with minimal dependencies
2. **Dependency Injection**: Components receive dependencies rather than creating them
3. **Event-Driven Communication**: Components communicate via events where possible
4. **Permission Boundaries**: All operations respect permission boundaries
5. **Safe Defaults**: All defaults are secure and require explicit opt-in for risky operations

## Key Components

### SystemOrchestrator

The central coordination component that manages the lifecycle of all other components and handles configuration, startup/shutdown, and error recovery.

**Key Responsibilities**:
- Component initialization in dependency order
- Configuration management
- Error handling and recovery
- System-wide event routing

### PermissionManager

Manages the permission system that controls what MacAgent can access.

**Key Responsibilities**:
- Permission management and enforcement
- Permission request handling
- Tracking and logging permission usage
- Temporary permission elevation

### SafetyProtocol

Ensures that potentially destructive operations are properly vetted.

**Key Responsibilities**:
- Risk assessment for operations
- Confirmation requests for risky operations
- Operation simulation
- Undo capability
- Emergency stop mechanism

### SecurityMonitor

Protects sensitive information and monitors for security risks.

**Key Responsibilities**:
- Credential secure storage
- Security event logging
- PII detection and redaction
- Suspicious activity detection

### WorkflowAutomator

Records, edits, and executes workflows across applications.

**Key Responsibilities**:
- Workflow recording
- Workflow execution
- Workflow editing
- Cross-application automation

### DiagnosticSystem

Provides system health monitoring and diagnostics.

**Key Responsibilities**:
- Component health monitoring
- System diagnostics
- Test execution
- Error reporting

## Development Environment Setup

### Prerequisites

- macOS 12.0 (Monterey) or later
- Python 3.9 or later
- Git
- Visual Studio Code (recommended) or other IDE

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/MacAgent.git
   cd MacAgent
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set Up Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

5. **Run Tests to Verify Setup**:
   ```bash
   pytest
   ```

### Development Commands

- **Run Tests**: `pytest`
- **Run Linting**: `flake8`
- **Run Type Checking**: `mypy src`
- **Generate Documentation**: `mkdocs build`
- **Run the Application in Development Mode**: `python -m MacAgent.app --dev`

## Project Structure

```
MacAgent/
├── MacAgent/                    # Main package
│   ├── __init__.py
│   ├── app.py                   # Application entry point
│   ├── src/                     # Source code
│   │   ├── core/                # Core systems
│   │   │   ├── system_orchestrator.py
│   │   │   └── ...
│   │   ├── intelligence/        # AI components
│   │   │   ├── workflow_automator.py
│   │   │   └── ...
│   │   ├── ui/                  # User interface components
│   │   │   ├── command_bar.py
│   │   │   └── ...
│   │   └── utils/               # Utility components
│   │       ├── permission_manager.py
│   │       ├── safety_protocol.py
│   │       ├── security_monitor.py
│   │       ├── diagnostic_system.py
│   │       └── ...
├── config/                      # Configuration files
├── docs/                        # Documentation
├── tests/                       # Tests
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── system/                  # System tests
├── scripts/                     # Utility scripts
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── requirements-dev.txt         # Development dependencies
```

## Security Architecture

### Multi-Layered Approach

MacAgent uses a multi-layered security architecture:

1. **Permission Layer**: Controls access to system resources
2. **Safety Layer**: Prevents accidental destructive operations
3. **Security Layer**: Protects sensitive information and detects risks
4. **Monitoring Layer**: Tracks and audits security-relevant events

### Permission System

The permission system is centered around the `PermissionManager` class:

```python
# Simplified example
permission_manager.request_permission(
    PermissionManager.FILE_READ,
    reason="Need to read configuration files",
    context={"path": "/Users/username/Documents/config.json"}
)
```

### Safety Protocols

The safety system prevents accidental destructive operations:

```python
# Simplified example
safety_protocol.safe_execute(
    SafetyProtocol.FILE_DELETE,
    "/Users/username/Documents/important.txt",
    os.remove,
    undo_function=restore_from_backup,
    description="Delete temporary file"
)
```

### Security Best Practices

When developing for MacAgent:

1. Always use the permission system for resource access
2. Use the safety protocol for potentially destructive operations
3. Never store credentials in plaintext
4. Always use the security monitor for handling sensitive data
5. Log security-relevant events appropriately

## Workflow System

The workflow system allows recording, editing, and executing multi-step workflows across applications.

### Workflow Structure

Workflows are stored as JSON:

```json
{
  "id": "workflow-123",
  "name": "Document Backup",
  "created_at": "2023-04-01T12:00:00Z",
  "modified_at": "2023-04-01T14:30:00Z",
  "steps": [
    {
      "app_name": "Finder",
      "action": "find_files",
      "parameters": {
        "location": "~/Documents",
        "modified_since": "7d"
      },
      "wait_time": 1.0
    },
    {
      "app_name": "Finder",
      "action": "compress_files",
      "parameters": {
        "output_path": "~/Backups/documents.zip"
      },
      "wait_time": null
    }
  ]
}
```

### Recording Workflows

Workflow recording captures user actions and translates them into reproducible steps:

```python
# Simplified example
workflow_id = workflow_automator.start_recording("Document Backup")

# Record steps (typically from UI actions)
workflow_automator.record_step(
    app_name="Finder",
    action="find_files",
    parameters={"location": "~/Documents", "modified_since": "7d"}
)

workflow_automator.record_step(
    app_name="Finder",
    action="compress_files",
    parameters={"output_path": "~/Backups/documents.zip"}
)

workflow = workflow_automator.stop_recording()
```

### Executing Workflows

Workflows are executed step by step:

```python
# Simplified example
result = workflow_automator.execute_workflow(
    "workflow-123",
    app_controller=app_controller
)
```

## API Documentation

### Core APIs

MacAgent exposes several core APIs for extensions and integrations.

#### SystemOrchestrator API

```python
# Get a component
component = system_orchestrator.get_component("permission_manager")

# Get component status
status = system_orchestrator.get_component_status("workflow_automator")

# Handle an error
system_orchestrator.handle_error(
    "security_monitor",
    error=Exception("Authentication failed"),
    context={"operation": "credential_validation"}
)
```

#### PermissionManager API

```python
# Request a permission
granted, message = permission_manager.request_permission(
    "file_read",
    reason="Reading configuration file",
    context={"path": "/path/to/file"}
)

# Check if a permission is granted
if permission_manager.check_permission("network_access"):
    # Perform network access

# Temporarily elevate a permission
permission_manager.elevate_permission_temporarily(
    "system_settings",
    duration_minutes=5,
    reason="Changing network settings"
)
```

#### WorkflowAutomator API

```python
# Start recording a workflow
workflow_id = workflow_automator.start_recording("Email Processing")

# Record a step
workflow_automator.record_step(
    app_name="Mail",
    action="filter_messages",
    parameters={"criteria": {"from": "important@example.com"}}
)

# Stop recording
workflow = workflow_automator.stop_recording()

# Execute a workflow
result = workflow_automator.execute_workflow(
    workflow_id,
    app_controller=app_controller
)
```

#### DiagnosticSystem API

```python
# Run all tests
results = diagnostic_system.run_all_tests()

# Generate a system report
report = diagnostic_system.generate_system_report(include_history=True)

# Check system health
health = diagnostic_system.check_system_health()
```

### Extension Points

MacAgent provides several extension points:

1. **Application Integrations**: Create new application integrations
2. **Workflow Actions**: Define custom workflow actions
3. **Command Handlers**: Add new commands to the command bar
4. **Security Extensions**: Extend the security system with new features

## Testing

MacAgent uses a comprehensive testing approach:

### Unit Tests

Test individual components in isolation. Each class and function should have unit tests:

```python
# Example unit test for PermissionManager
def test_request_permission():
    permission_manager = PermissionManager()
    granted, _ = permission_manager.request_permission(
        "file_read",
        reason="Testing"
    )
    assert granted == True  # Low-risk permissions are auto-granted in tests
```

### Integration Tests

Test how components interact:

```python
# Example integration test
def test_workflow_with_permissions():
    # Set up components
    permission_manager = PermissionManager()
    workflow_automator = WorkflowAutomator()
    
    # Test their interaction
    workflow_id = workflow_automator.start_recording("Test")
    # ...record steps...
    workflow = workflow_automator.stop_recording()
    
    # Execute with permission checks
    result = workflow_automator.execute_workflow(
        workflow_id,
        app_controller=MockAppController(permission_manager)
    )
    assert result["success"] == True
```

### System Tests

Test the entire system:

```python
# Example system test
def test_end_to_end_workflow():
    # Start the application with test configuration
    app = start_test_app()
    
    # Simulate user creating a workflow
    app.execute_command("workflow create test-workflow")
    # ...simulate more user actions...
    
    # Verify the outcome
    assert "test-workflow" in app.execute_command("workflow list")
```

### Running Tests

- Run all tests: `pytest`
- Run unit tests: `pytest tests/unit`
- Run integration tests: `pytest tests/integration`
- Run system tests: `pytest tests/system`
- Generate coverage report: `pytest --cov=MacAgent`

## Contributing

We welcome contributions to MacAgent! Here's how to contribute:

### Contribution Workflow

1. **Find or Create an Issue**: Start with an issue on GitHub
2. **Fork the Repository**: Create your own fork
3. **Create a Branch**: `git checkout -b feature/your-feature-name`
4. **Make Changes**: Implement your changes
5. **Write Tests**: Ensure your changes are covered by tests
6. **Check Style**: Run linters and formatters
7. **Submit a Pull Request**: Create a PR with a clear description

### Pull Request Requirements

- All tests must pass
- Code must be properly documented
- Changes must follow the style guide
- Security-sensitive changes require additional review

### Code Review Process

1. Automated checks run on all PRs
2. At least one maintainer must review and approve
3. Security-related changes require review by a security team member
4. Changes may require revisions before merging

## Style Guide

MacAgent follows a consistent style guide:

### Python Code Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use type hints for function signatures
- Document all classes and functions with docstrings
- Keep functions focused and small (under 50 lines when possible)
- Use meaningful variable and function names

### Documentation Style

- Use Markdown for documentation
- Include examples for all public APIs
- Keep documentation up to date with code changes
- Use diagrams to explain complex concepts

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat(component): add new feature
fix(component): fix issue in component
docs(component): update documentation
test(component): add or update tests
refactor(component): code refactoring without changes
```

## Release Process

MacAgent follows a structured release process:

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Incompatible API changes
- **Minor** (0.x.0): Backwards-compatible new features
- **Patch** (0.0.x): Backwards-compatible bug fixes

### Release Steps

1. **Prepare Release**: Create a release branch
2. **Update Version**: Update version numbers in code
3. **Update Changelog**: Document changes since the last release
4. **Create Release Candidate**: Tag a release candidate
5. **Test Release Candidate**: Run comprehensive tests
6. **Create Final Release**: Tag the final release
7. **Publish Release**: Publish packages and update documentation

### Hotfix Process

For critical issues between releases:

1. Create a hotfix branch from the release tag
2. Fix the issue
3. Create a new patch release
4. Cherry-pick changes to the development branch

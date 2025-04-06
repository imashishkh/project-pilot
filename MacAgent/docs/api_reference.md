# MacAgent API Reference

This reference documentation provides details about the main classes, methods, and interfaces available in MacAgent.

## Table of Contents

- [Core APIs](#core-apis)
  - [SystemOrchestrator](#systemorchestrator)
  - [PermissionManager](#permissionmanager)
  - [SafetyProtocol](#safetyprotocol)
  - [SecurityMonitor](#securitymonitor)
  - [WorkflowAutomator](#workflowautomator)
  - [DiagnosticSystem](#diagnosticsystem)
- [UI APIs](#ui-apis)
  - [CommandBar](#commandbar)
  - [NotificationCenter](#notificationcenter)
- [Utility APIs](#utility-apis)
  - [ConfigManager](#configmanager)
  - [Logger](#logger)

## Core APIs

### SystemOrchestrator

The central component that manages the lifecycle of all other components.

```python
class SystemOrchestrator:
    def __init__(
        self, 
        config_path: str = None, 
        logs_path: str = None, 
        components: Dict[str, Any] = None
    ):
        """
        Initialize the system orchestrator.
        
        Args:
            config_path: Path to the configuration directory
            logs_path: Path to the logs directory
            components: Dictionary of components to initialize
        """
```

#### Methods

##### `initialize_components()`

```python
def initialize_components(self) -> bool:
    """
    Initialize all components in the correct dependency order.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
```

##### `get_component()`

```python
def get_component(self, component_name: str) -> Any:
    """
    Get a component by name.
    
    Args:
        component_name: Name of the component to retrieve
        
    Returns:
        The requested component instance or None if not found
        
    Raises:
        ComponentNotFoundError: If the component doesn't exist
    """
```

##### `get_component_status()`

```python
def get_component_status(self, component_name: str) -> Dict[str, Any]:
    """
    Get the status of a component.
    
    Args:
        component_name: Name of the component to check
        
    Returns:
        Dictionary containing component status
        
    Raises:
        ComponentNotFoundError: If the component doesn't exist
    """
```

##### `handle_error()`

```python
def handle_error(
    self, 
    component_name: str, 
    error: Exception, 
    context: Dict[str, Any] = None
) -> None:
    """
    Handle an error from a component.
    
    Args:
        component_name: Name of the component that raised the error
        error: The exception that was raised
        context: Additional context about the error
    """
```

##### `shutdown()`

```python
def shutdown(self, timeout: int = 30) -> bool:
    """
    Shut down all components gracefully.
    
    Args:
        timeout: Maximum time in seconds to wait for components to shut down
        
    Returns:
        bool: True if shutdown was successful, False otherwise
    """
```

#### Examples

```python
# Initialize the orchestrator
orchestrator = SystemOrchestrator(
    config_path="/Users/username/Library/Application Support/MacAgent/config",
    logs_path="/Users/username/Library/Logs/MacAgent"
)

# Initialize components
success = orchestrator.initialize_components()
if not success:
    print("Failed to initialize components")
    
# Get a component
permission_manager = orchestrator.get_component("permission_manager")

# Check component status
status = orchestrator.get_component_status("workflow_automator")
if status["status"] == "error":
    print(f"Workflow automator error: {status['message']}")
    
# Handle an error
try:
    # Some operation
    pass
except Exception as e:
    orchestrator.handle_error(
        "security_monitor", 
        e, 
        {"operation": "credential_validation"}
    )
    
# Shutdown the system
orchestrator.shutdown()
```

### PermissionManager

Manages access permissions for the agent.

```python
class PermissionManager:
    # Permission types
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK_ACCESS = "network_access"
    SYSTEM_SETTINGS = "system_settings"
    NOTIFICATION = "notification"
    CLIPBOARD = "clipboard"
    
    def __init__(
        self, 
        config_path: str = None, 
        logs_path: str = None
    ):
        """
        Initialize the permission manager.
        
        Args:
            config_path: Path to the configuration directory
            logs_path: Path to the logs directory
        """
```

#### Methods

##### `request_permission()`

```python
def request_permission(
    self, 
    permission_type: str, 
    reason: str = None, 
    context: Dict[str, Any] = None,
    auto_grant: bool = False
) -> Tuple[bool, str]:
    """
    Request a permission.
    
    Args:
        permission_type: Type of permission requested
        reason: Reason for requesting the permission
        context: Additional context about the request
        auto_grant: Whether to automatically grant low-risk permissions
        
    Returns:
        Tuple of (granted: bool, message: str)
    """
```

##### `check_permission()`

```python
def check_permission(
    self, 
    permission_type: str, 
    context: Dict[str, Any] = None
) -> bool:
    """
    Check if a permission is granted.
    
    Args:
        permission_type: Type of permission to check
        context: Additional context about the check
        
    Returns:
        bool: True if the permission is granted, False otherwise
    """
```

##### `elevate_permission_temporarily()`

```python
def elevate_permission_temporarily(
    self, 
    permission_type: str, 
    duration_minutes: int = 5, 
    reason: str = None
) -> Tuple[bool, str]:
    """
    Temporarily elevate a permission.
    
    Args:
        permission_type: Type of permission to elevate
        duration_minutes: Duration of elevation in minutes
        reason: Reason for elevating the permission
        
    Returns:
        Tuple of (success: bool, message: str)
    """
```

##### `revoke_permission()`

```python
def revoke_permission(
    self, 
    permission_type: str, 
    context: Dict[str, Any] = None
) -> bool:
    """
    Revoke a previously granted permission.
    
    Args:
        permission_type: Type of permission to revoke
        context: Additional context about the revocation
        
    Returns:
        bool: True if the permission was revoked, False otherwise
    """
```

#### Examples

```python
# Initialize the permission manager
permission_manager = PermissionManager()

# Request permission to read a file
granted, message = permission_manager.request_permission(
    PermissionManager.FILE_READ,
    reason="Need to read configuration files",
    context={"path": "/Users/username/Documents/config.json"}
)

if granted:
    # Proceed with reading the file
    with open("/Users/username/Documents/config.json", "r") as f:
        config = json.load(f)
else:
    print(f"Permission denied: {message}")
    
# Check if network access is permitted
if permission_manager.check_permission(PermissionManager.NETWORK_ACCESS):
    # Perform network request
    response = requests.get("https://api.example.com")
    
# Temporarily elevate permissions
success, message = permission_manager.elevate_permission_temporarily(
    PermissionManager.SYSTEM_SETTINGS,
    duration_minutes=5,
    reason="Changing network settings"
)

# Revoke a permission
permission_manager.revoke_permission(PermissionManager.CLIPBOARD)
```

### SafetyProtocol

Ensures that potentially destructive operations are properly vetted.

```python
class SafetyProtocol:
    # Operation types
    FILE_DELETE = "file_delete"
    FILE_MODIFY = "file_modify"
    SYSTEM_MODIFY = "system_modify"
    NETWORK_REQUEST = "network_request"
    
    def __init__(
        self, 
        config_path: str = None, 
        permission_manager: PermissionManager = None
    ):
        """
        Initialize the safety protocol.
        
        Args:
            config_path: Path to the configuration directory
            permission_manager: PermissionManager instance
        """
```

#### Methods

##### `assess_risk()`

```python
def assess_risk(
    self, 
    operation_type: str, 
    target: str, 
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Assess the risk of an operation.
    
    Args:
        operation_type: Type of operation
        target: Target of the operation
        details: Additional details about the operation
        
    Returns:
        Dictionary containing risk assessment
    """
```

##### `safe_execute()`

```python
def safe_execute(
    self, 
    operation_type: str, 
    target: str, 
    operation_function: Callable, 
    undo_function: Callable = None, 
    description: str = None, 
    auto_confirm: bool = False
) -> Dict[str, Any]:
    """
    Safely execute an operation with risk assessment.
    
    Args:
        operation_type: Type of operation
        target: Target of the operation
        operation_function: Function to execute the operation
        undo_function: Function to undo the operation (if possible)
        description: Description of the operation
        auto_confirm: Whether to automatically confirm low-risk operations
        
    Returns:
        Dictionary containing operation result
    """
```

##### `simulate_operation()`

```python
def simulate_operation(
    self, 
    operation_type: str, 
    target: str, 
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Simulate an operation without executing it.
    
    Args:
        operation_type: Type of operation
        target: Target of the operation
        details: Additional details about the operation
        
    Returns:
        Dictionary containing simulation result
    """
```

##### `emergency_stop()`

```python
def emergency_stop(self) -> bool:
    """
    Trigger an emergency stop of all operations.
    
    Returns:
        bool: True if emergency stop was successful, False otherwise
    """
```

#### Examples

```python
# Initialize the safety protocol
safety_protocol = SafetyProtocol(
    permission_manager=permission_manager
)

# Assess the risk of an operation
risk = safety_protocol.assess_risk(
    SafetyProtocol.FILE_DELETE,
    "/Users/username/Documents/important.txt",
    {"reason": "Temporary file cleanup"}
)

if risk["level"] > 7:
    print(f"High risk operation: {risk['explanation']}")

# Define an undo function for backup
def restore_from_backup(file_path):
    shutil.copy(f"{file_path}.backup", file_path)

# Safely execute an operation
result = safety_protocol.safe_execute(
    SafetyProtocol.FILE_DELETE,
    "/Users/username/Documents/temp.txt",
    os.remove,
    undo_function=restore_from_backup,
    description="Delete temporary file"
)

if not result["success"]:
    print(f"Operation failed: {result['message']}")

# Simulate an operation
simulation = safety_protocol.simulate_operation(
    SafetyProtocol.SYSTEM_MODIFY,
    "network_settings",
    {"changes": {"dns_servers": ["8.8.8.8", "8.8.4.4"]}}
)

print(f"Simulation results: {simulation['effects']}")

# Emergency stop
if dangerous_condition:
    safety_protocol.emergency_stop()
```

### SecurityMonitor

Protects sensitive information and monitors for security risks.

```python
class SecurityMonitor:
    def __init__(
        self, 
        config_path: str = None, 
        logs_path: str = None
    ):
        """
        Initialize the security monitor.
        
        Args:
            config_path: Path to the configuration directory
            logs_path: Path to the logs directory
        """
```

#### Methods

##### `store_credential()`

```python
def store_credential(
    self, 
    credential_id: str, 
    credential_value: str, 
    description: str = None, 
    expiry: datetime = None
) -> bool:
    """
    Securely store a credential.
    
    Args:
        credential_id: Identifier for the credential
        credential_value: Value of the credential
        description: Description of the credential
        expiry: Optional expiry date for the credential
        
    Returns:
        bool: True if the credential was stored, False otherwise
    """
```

##### `retrieve_credential()`

```python
def retrieve_credential(
    self, 
    credential_id: str
) -> str:
    """
    Retrieve a stored credential.
    
    Args:
        credential_id: Identifier for the credential
        
    Returns:
        The credential value
        
    Raises:
        CredentialNotFoundError: If the credential doesn't exist
        CredentialExpiredError: If the credential has expired
    """
```

##### `detect_pii()`

```python
def detect_pii(
    self, 
    text: str
) -> Dict[str, Any]:
    """
    Detect personally identifiable information in text.
    
    Args:
        text: Text to scan for PII
        
    Returns:
        Dictionary containing detected PII types and positions
    """
```

##### `redact_pii()`

```python
def redact_pii(
    self, 
    text: str, 
    replacement: str = "[REDACTED]"
) -> str:
    """
    Redact personally identifiable information from text.
    
    Args:
        text: Text to redact PII from
        replacement: Text to replace PII with
        
    Returns:
        Redacted text
    """
```

##### `log_security_event()`

```python
def log_security_event(
    self, 
    event_type: str, 
    details: Dict[str, Any], 
    severity: str = "info"
) -> None:
    """
    Log a security-related event.
    
    Args:
        event_type: Type of security event
        details: Details about the event
        severity: Severity level (info, warning, error, critical)
    """
```

#### Examples

```python
# Initialize the security monitor
security_monitor = SecurityMonitor()

# Store API credentials
success = security_monitor.store_credential(
    "api_key_example",
    "sk_live_1234567890abcdef",
    description="Example API Key",
    expiry=datetime.now() + timedelta(days=90)
)

# Retrieve stored credentials
try:
    api_key = security_monitor.retrieve_credential("api_key_example")
    # Use the API key
except CredentialNotFoundError:
    print("API key not found")
except CredentialExpiredError:
    print("API key has expired")

# Detect PII in text
user_input = "My name is John Doe and my SSN is 123-45-6789"
pii_detected = security_monitor.detect_pii(user_input)
if pii_detected["found"]:
    print(f"Found PII: {', '.join(pii_detected['types'])}")

# Redact PII from text
redacted_text = security_monitor.redact_pii(user_input)
print(redacted_text)  # "My name is [REDACTED] and my SSN is [REDACTED]"

# Log security events
security_monitor.log_security_event(
    "failed_login_attempt",
    {"username": "user123", "ip_address": "192.168.1.1", "attempt": 3},
    severity="warning"
)
```

### WorkflowAutomator

Records, edits, and executes workflows across applications.

```python
class WorkflowAutomator:
    def __init__(
        self, 
        config_path: str = None, 
        workflows_path: str = None, 
        permission_manager: PermissionManager = None, 
        safety_protocol: SafetyProtocol = None
    ):
        """
        Initialize the workflow automator.
        
        Args:
            config_path: Path to the configuration directory
            workflows_path: Path to store workflows
            permission_manager: PermissionManager instance
            safety_protocol: SafetyProtocol instance
        """
```

#### Methods

##### `start_recording()`

```python
def start_recording(self, workflow_name: str) -> str:
    """
    Start recording a new workflow.
    
    Args:
        workflow_name: Name for the new workflow
        
    Returns:
        str: Workflow ID
    """
```

##### `record_step()`

```python
def record_step(
    self, 
    app_name: str, 
    action: str, 
    parameters: Dict[str, Any] = None, 
    wait_time: float = None
) -> bool:
    """
    Record a step in the current workflow.
    
    Args:
        app_name: Name of the application
        action: Action to perform
        parameters: Parameters for the action
        wait_time: Time to wait after action (in seconds)
        
    Returns:
        bool: True if the step was recorded, False otherwise
    """
```

##### `stop_recording()`

```python
def stop_recording(self) -> Dict[str, Any]:
    """
    Stop recording the current workflow.
    
    Returns:
        Dictionary containing the recorded workflow
    """
```

##### `get_workflow()`

```python
def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
    """
    Get a workflow by ID.
    
    Args:
        workflow_id: ID of the workflow to retrieve
        
    Returns:
        Dictionary containing the workflow
        
    Raises:
        WorkflowNotFoundError: If the workflow doesn't exist
    """
```

##### `list_workflows()`

```python
def list_workflows(self) -> List[Dict[str, Any]]:
    """
    List all available workflows.
    
    Returns:
        List of workflows with basic information
    """
```

##### `execute_workflow()`

```python
def execute_workflow(
    self, 
    workflow_id: str, 
    app_controller: AppController, 
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute a workflow.
    
    Args:
        workflow_id: ID of the workflow to execute
        app_controller: AppController instance for interacting with applications
        parameters: Optional parameters to override workflow values
        
    Returns:
        Dictionary containing execution results
    """
```

##### `edit_workflow()`

```python
def edit_workflow(
    self, 
    workflow_id: str, 
    changes: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Edit an existing workflow.
    
    Args:
        workflow_id: ID of the workflow to edit
        changes: Changes to apply to the workflow
        
    Returns:
        Dictionary containing the updated workflow
    """
```

#### Examples

```python
# Initialize the workflow automator
workflow_automator = WorkflowAutomator(
    permission_manager=permission_manager,
    safety_protocol=safety_protocol
)

# Start recording a workflow
workflow_id = workflow_automator.start_recording("Document Backup")

# Record workflow steps
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

# Stop recording and save workflow
workflow = workflow_automator.stop_recording()
print(f"Recorded workflow: {workflow['name']} with ID {workflow['id']}")

# List all available workflows
workflows = workflow_automator.list_workflows()
for wf in workflows:
    print(f"{wf['id']}: {wf['name']} ({len(wf['steps'])} steps)")

# Execute a workflow
result = workflow_automator.execute_workflow(
    workflow_id,
    app_controller=app_controller,
    parameters={"output_path": "~/Backups/documents_custom.zip"}
)

if result["success"]:
    print("Workflow executed successfully")
else:
    print(f"Workflow execution failed: {result['error']}")

# Edit an existing workflow
updated_workflow = workflow_automator.edit_workflow(
    workflow_id,
    {"name": "Weekly Document Backup", "steps": [...]}
)
```

### DiagnosticSystem

Provides system health monitoring and diagnostics.

```python
class DiagnosticSystem:
    def __init__(
        self, 
        system_orchestrator: SystemOrchestrator = None, 
        config_path: str = None, 
        logs_path: str = None,
        check_interval_minutes: int = 30
    ):
        """
        Initialize the diagnostic system.
        
        Args:
            system_orchestrator: SystemOrchestrator instance
            config_path: Path to the configuration directory
            logs_path: Path to the logs directory
            check_interval_minutes: Interval between health checks in minutes
        """
```

#### Methods

##### `register_test()`

```python
def register_test(
    self, 
    test_name: str, 
    test_function: Callable, 
    component: str = None, 
    critical: bool = False, 
    description: str = None
) -> bool:
    """
    Register a system test.
    
    Args:
        test_name: Name of the test
        test_function: Function that performs the test
        component: Component being tested
        critical: Whether the test is critical for system operation
        description: Description of what the test checks
        
    Returns:
        bool: True if the test was registered, False otherwise
    """
```

##### `run_test()`

```python
def run_test(self, test_name: str) -> Dict[str, Any]:
    """
    Run a specific test.
    
    Args:
        test_name: Name of the test to run
        
    Returns:
        Dictionary containing test results
        
    Raises:
        TestNotFoundError: If the test doesn't exist
    """
```

##### `run_all_tests()`

```python
def run_all_tests(
    self, 
    component: str = None, 
    critical_only: bool = False
) -> Dict[str, Any]:
    """
    Run all tests or tests for a specific component.
    
    Args:
        component: Component to run tests for (None for all)
        critical_only: Whether to run only critical tests
        
    Returns:
        Dictionary containing test results
    """
```

##### `check_system_health()`

```python
def check_system_health(self) -> Dict[str, Any]:
    """
    Check overall system health.
    
    Returns:
        Dictionary containing health status
    """
```

##### `check_resource_usage()`

```python
def check_resource_usage(self) -> Dict[str, Any]:
    """
    Check system resource usage.
    
    Returns:
        Dictionary containing resource usage information
    """
```

##### `generate_system_report()`

```python
def generate_system_report(
    self, 
    include_history: bool = True, 
    include_config: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive system report.
    
    Args:
        include_history: Whether to include historical data
        include_config: Whether to include configuration data
        
    Returns:
        Dictionary containing the system report
    """
```

#### Examples

```python
# Initialize the diagnostic system
diagnostic_system = DiagnosticSystem(
    system_orchestrator=orchestrator,
    check_interval_minutes=15
)

# Register a custom test
def check_database_connection():
    try:
        db.connect()
        return True
    except:
        return False

diagnostic_system.register_test(
    "database_connection",
    check_database_connection,
    component="database",
    critical=True,
    description="Checks if the database connection is working"
)

# Run a specific test
result = diagnostic_system.run_test("database_connection")
if not result["passed"]:
    print(f"Database connection test failed: {result['error']}")

# Run all tests for a component
results = diagnostic_system.run_all_tests(component="database")
failed_tests = [t for t in results["tests"] if not t["passed"]]
if failed_tests:
    print(f"Failed {len(failed_tests)} database tests")

# Check system health
health = diagnostic_system.check_system_health()
if health["status"] != "healthy":
    print(f"System health issues: {health['issues']}")

# Check resource usage
resources = diagnostic_system.check_resource_usage()
print(f"CPU: {resources['cpu']}%, Memory: {resources['memory']}%")

# Generate a system report
report = diagnostic_system.generate_system_report(include_history=True)
print(f"System report generated with {len(report['tests'])} tests")
```

## UI APIs

### CommandBar

Provides a command interface for users to interact with MacAgent.

```python
class CommandBar:
    def __init__(
        self, 
        system_orchestrator: SystemOrchestrator = None
    ):
        """
        Initialize the command bar.
        
        Args:
            system_orchestrator: SystemOrchestrator instance
        """
```

#### Methods

##### `register_command()`

```python
def register_command(
    self, 
    command_name: str, 
    handler: Callable, 
    description: str = None, 
    examples: List[str] = None
) -> bool:
    """
    Register a command handler.
    
    Args:
        command_name: Name of the command
        handler: Function that handles the command
        description: Description of the command
        examples: Example usages of the command
        
    Returns:
        bool: True if the command was registered, False otherwise
    """
```

##### `execute_command()`

```python
def execute_command(
    self, 
    command_text: str
) -> Any:
    """
    Execute a command.
    
    Args:
        command_text: Text of the command to execute
        
    Returns:
        Result of the command
        
    Raises:
        CommandNotFoundError: If the command doesn't exist
        CommandExecutionError: If the command execution fails
    """
```

##### `suggest_commands()`

```python
def suggest_commands(
    self, 
    partial_text: str, 
    limit: int = 5
) -> List[Dict[str, str]]:
    """
    Suggest commands based on partial text.
    
    Args:
        partial_text: Partial command text
        limit: Maximum number of suggestions
        
    Returns:
        List of command suggestions
    """
```

#### Examples

```python
# Initialize the command bar
command_bar = CommandBar(system_orchestrator=orchestrator)

# Register a command
def handle_backup_command(args):
    source = args.get("source", "~/Documents")
    destination = args.get("destination", "~/Backups")
    return {"success": True, "files_backed_up": 42}

command_bar.register_command(
    "backup",
    handle_backup_command,
    description="Backup files from source to destination",
    examples=["backup", "backup --source ~/Work --destination ~/Backups/Work"]
)

# Execute a command
result = command_bar.execute_command("backup --source ~/Documents/Projects")
if result["success"]:
    print(f"Backed up {result['files_backed_up']} files")

# Get command suggestions
suggestions = command_bar.suggest_commands("back")
for suggestion in suggestions:
    print(f"{suggestion['command']}: {suggestion['description']}")
```

### NotificationCenter

Manages notifications to the user.

```python
class NotificationCenter:
    # Notification priorities
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __init__(
        self, 
        permission_manager: PermissionManager = None
    ):
        """
        Initialize the notification center.
        
        Args:
            permission_manager: PermissionManager instance
        """
```

#### Methods

##### `send_notification()`

```python
def send_notification(
    self, 
    title: str, 
    message: str, 
    priority: str = "normal", 
    actions: List[Dict[str, Any]] = None,
    icon: str = None
) -> bool:
    """
    Send a notification to the user.
    
    Args:
        title: Title of the notification
        message: Message content
        priority: Priority level
        actions: List of actions the user can take
        icon: Path to notification icon
        
    Returns:
        bool: True if the notification was sent, False otherwise
    """
```

##### `schedule_notification()`

```python
def schedule_notification(
    self, 
    title: str, 
    message: str, 
    schedule_time: datetime, 
    priority: str = "normal", 
    actions: List[Dict[str, Any]] = None,
    icon: str = None
) -> str:
    """
    Schedule a notification for later delivery.
    
    Args:
        title: Title of the notification
        message: Message content
        schedule_time: When to send the notification
        priority: Priority level
        actions: List of actions the user can take
        icon: Path to notification icon
        
    Returns:
        str: ID of the scheduled notification
    """
```

##### `cancel_notification()`

```python
def cancel_notification(self, notification_id: str) -> bool:
    """
    Cancel a scheduled notification.
    
    Args:
        notification_id: ID of the notification to cancel
        
    Returns:
        bool: True if the notification was cancelled, False otherwise
    """
```

#### Examples

```python
# Initialize the notification center
notification_center = NotificationCenter(
    permission_manager=permission_manager
)

# Send a notification
result = notification_center.send_notification(
    title="Backup Complete",
    message="Your files have been backed up successfully",
    priority=NotificationCenter.NORMAL,
    actions=[
        {"title": "View Backup", "command": "open ~/Backups"},
        {"title": "Dismiss", "command": None}
    ],
    icon="backup_icon.png"
)

# Schedule a notification
tomorrow = datetime.now() + timedelta(days=1)
notification_id = notification_center.schedule_notification(
    title="Weekly Cleanup Reminder",
    message="Time to clean up your Downloads folder",
    schedule_time=tomorrow,
    priority=NotificationCenter.LOW
)

# Cancel a scheduled notification
notification_center.cancel_notification(notification_id)
```

## Utility APIs

### ConfigManager

Manages configuration settings for MacAgent.

```python
class ConfigManager:
    def __init__(
        self, 
        config_path: str = None, 
        default_config: Dict[str, Any] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration directory
            default_config: Default configuration if none exists
        """
```

#### Methods

##### `get_config()`

```python
def get_config(
    self, 
    component: str = None, 
    create_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Get configuration for a component or the entire system.
    
    Args:
        component: Component to get configuration for (None for all)
        create_if_missing: Whether to create default config if missing
        
    Returns:
        Dictionary containing configuration
    """
```

##### `update_config()`

```python
def update_config(
    self, 
    updates: Dict[str, Any], 
    component: str = None
) -> bool:
    """
    Update configuration settings.
    
    Args:
        updates: Dictionary of updates to apply
        component: Component to update (None for main config)
        
    Returns:
        bool: True if the configuration was updated, False otherwise
    """
```

##### `reset_config()`

```python
def reset_config(
    self, 
    component: str = None
) -> bool:
    """
    Reset configuration to defaults.
    
    Args:
        component: Component to reset (None for all)
        
    Returns:
        bool: True if the configuration was reset, False otherwise
    """
```

#### Examples

```python
# Initialize the configuration manager
config_manager = ConfigManager()

# Get configuration
config = config_manager.get_config()
workflow_config = config_manager.get_config("workflow_automator")

# Update configuration
config_manager.update_config(
    {"check_interval_minutes": 15},
    component="diagnostic_system"
)

# Reset configuration to defaults
config_manager.reset_config("permission_manager")
```

### Logger

Provides logging capabilities throughout the application.

```python
class Logger:
    # Log levels
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def __init__(
        self, 
        name: str, 
        logs_path: str = None, 
        log_level: str = "info", 
        max_size_mb: int = 10, 
        backup_count: int = 5
    ):
        """
        Initialize a logger.
        
        Args:
            name: Name of the logger
            logs_path: Path to store log files
            log_level: Minimum level to log
            max_size_mb: Maximum size of log files in MB
            backup_count: Number of backup files to keep
        """
```

#### Methods

##### `log()`

```python
def log(
    self, 
    level: str, 
    message: str, 
    context: Dict[str, Any] = None
) -> None:
    """
    Log a message.
    
    Args:
        level: Log level
        message: Message to log
        context: Additional context data
    """
```

##### `debug()`

```python
def debug(
    self, 
    message: str, 
    context: Dict[str, Any] = None
) -> None:
    """
    Log a debug message.
    
    Args:
        message: Message to log
        context: Additional context data
    """
```

##### `info()`

```python
def info(
    self, 
    message: str, 
    context: Dict[str, Any] = None
) -> None:
    """
    Log an info message.
    
    Args:
        message: Message to log
        context: Additional context data
    """
```

##### `warning()`

```python
def warning(
    self, 
    message: str, 
    context: Dict[str, Any] = None
) -> None:
    """
    Log a warning message.
    
    Args:
        message: Message to log
        context: Additional context data
    """
```

##### `error()`

```python
def error(
    self, 
    message: str, 
    context: Dict[str, Any] = None
) -> None:
    """
    Log an error message.
    
    Args:
        message: Message to log
        context: Additional context data
    """
```

##### `critical()`

```python
def critical(
    self, 
    message: str, 
    context: Dict[str, Any] = None
) -> None:
    """
    Log a critical message.
    
    Args:
        message: Message to log
        context: Additional context data
    """
```

#### Examples

```python
# Initialize a logger
logger = Logger(
    name="workflow_automator",
    logs_path="/Users/username/Library/Logs/MacAgent",
    log_level=Logger.INFO
)

# Log messages at different levels
logger.debug("Starting workflow execution")

logger.info(
    "Workflow executed successfully",
    {"workflow_id": "backup-123", "steps_completed": 5}
)

logger.warning(
    "Step execution time exceeded threshold",
    {"step": 3, "execution_time": 12.5, "threshold": 10.0}
)

logger.error(
    "Failed to execute workflow step",
    {"workflow_id": "backup-123", "step": 4, "error": "Permission denied"}
)

logger.critical(
    "Critical system failure",
    {"component": "database", "error": "Connection lost"}
)

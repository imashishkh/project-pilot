# MacAgent User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Features](#features)
5. [Command Reference](#command-reference)
6. [Workflows](#workflows)
7. [Security and Permissions](#security-and-permissions)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Support](#support)

## Introduction

MacAgent is an AI-powered assistant designed specifically for macOS that helps automate tasks, enhance productivity, and simplify complex operations on your Mac. With advanced security features, workflow automation capabilities, and an intuitive interface, MacAgent provides a secure and intelligent companion for your daily computing needs.

### Key Benefits

- **Automation**: Automate repetitive tasks and complex workflows
- **Enhanced Security**: Multi-layered security system with granular permissions
- **Intelligent Assistance**: Context-aware help and suggestions
- **Cross-Application Support**: Work seamlessly across different applications
- **Privacy-First Design**: Your data stays on your machine

## Installation

### System Requirements

- macOS 12.0 (Monterey) or later
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- Python 3.9 or later

### Installation Steps

1. **Download MacAgent**:
   ```bash
   git clone https://github.com/yourusername/MacAgent.git
   cd MacAgent
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the application**:
   ```bash
   python setup.py
   ```

5. **Start MacAgent**:
   ```bash
   python -m MacAgent.app
   ```

### First-time Configuration

When you first start MacAgent, you'll be guided through a setup process to:

1. Configure basic settings
2. Set up security preferences
3. Grant necessary permissions
4. Create your first workflow (optional)
5. Customize your experience

## Getting Started

### The MacAgent Interface

The MacAgent interface consists of:

- **Command Bar**: For entering commands and queries
- **Status Panel**: Displays system status and active components
- **Workflow Panel**: Manage and monitor your workflows
- **Output Area**: View results and feedback from commands

### Basic Commands

Here are some basic commands to get you started:

- `help`: Display available commands and help information
- `status`: Check the status of MacAgent components
- `workflow list`: Show all available workflows
- `workflow run [name]`: Run a specific workflow
- `exit`: Close MacAgent

### Quick Tutorial

1. **Ask MacAgent for help**:
   ```
   help
   ```

2. **Check system status**:
   ```
   status
   ```

3. **Create your first workflow**:
   ```
   workflow create
   ```
   Follow the on-screen prompts to create a simple workflow.

4. **Run your workflow**:
   ```
   workflow run my-first-workflow
   ```

## Features

### Workflow Automation

MacAgent can automate multi-step workflows across applications:

1. **Record workflows**: Capture your actions to create reusable workflows
2. **Edit workflows**: Modify existing workflows through the interface
3. **Schedule workflows**: Run workflows at specific times or triggers
4. **Share workflows**: Export and import workflows with other MacAgent users

### Cross-Application Integration

MacAgent works with popular macOS applications:

- Finder for file management
- Mail for email automation
- Calendar for appointment scheduling
- Terminal for command execution
- And many more system applications

### Security Features

MacAgent includes robust security features:

- **Permission Management**: Control what MacAgent can access
- **Safety Protocols**: Confirm potentially destructive operations
- **Security Monitoring**: Detect and prevent security risks
- **Encrypted Storage**: Securely store sensitive information

### Contextual Help

Get help based on your current context:

- **Smart Suggestions**: Recommendations based on your usage patterns
- **Guided Troubleshooting**: Step-by-step help for common issues
- **Task Completion**: Suggestions to complete your current task

## Command Reference

### General Commands

| Command | Description | Example |
|---------|-------------|---------|
| `help [topic]` | Display help for a specific topic | `help workflows` |
| `status` | Show system status | `status` |
| `exit` | Exit MacAgent | `exit` |
| `version` | Display version information | `version` |

### Workflow Commands

| Command | Description | Example |
|---------|-------------|---------|
| `workflow list` | List all workflows | `workflow list` |
| `workflow create` | Create a new workflow | `workflow create` |
| `workflow run [name]` | Run a workflow | `workflow run backup-documents` |
| `workflow edit [name]` | Edit a workflow | `workflow edit backup-documents` |
| `workflow delete [name]` | Delete a workflow | `workflow delete backup-documents` |
| `workflow export [name]` | Export a workflow | `workflow export backup-documents` |
| `workflow import [file]` | Import a workflow | `workflow import workflow.json` |

### Security Commands

| Command | Description | Example |
|---------|-------------|---------|
| `security status` | View security status | `security status` |
| `permissions list` | List current permissions | `permissions list` |
| `permissions grant [name]` | Grant a permission | `permissions grant file_read` |
| `permissions revoke [name]` | Revoke a permission | `permissions revoke file_write` |

## Workflows

### Creating a Workflow

To create a new workflow:

1. Enter `workflow create` in the command bar
2. Provide a name and description for your workflow
3. Choose whether to record actions or build step-by-step
4. Follow the prompts to add steps to your workflow
5. Test and save your workflow

### Example Workflows

#### Document Backup Workflow

This workflow automatically backs up documents to a specified location:

1. Collect all modified documents from the past week
2. Compress documents into a zip file
3. Move the zip file to a backup location
4. Send a notification that backup is complete

Command:
```
workflow run document-backup
```

#### Email Processing Workflow

Automatically organize and process emails:

1. Check for new emails in the inbox
2. Filter emails based on criteria (sender, subject, etc.)
3. Move emails to appropriate folders
4. Generate a summary report

Command:
```
workflow run process-emails
```

## Security and Permissions

### Permission Levels

MacAgent uses a granular permission system:

- **File Access**: Read, write, and delete files
- **Network Access**: Connect to internet resources
- **System Commands**: Execute system commands
- **Application Control**: Interact with other applications

### Managing Permissions

View and manage permissions with these commands:

- `permissions list`: View all permissions and their status
- `permissions grant [name]`: Grant a specific permission
- `permissions revoke [name]`: Revoke a specific permission
- `permissions explain [name]`: Get details about a permission

### Security Best Practices

1. Only grant permissions that are necessary
2. Regularly review granted permissions
3. Use temporary permission elevation for one-time tasks
4. Keep MacAgent updated to the latest version
5. Review security logs periodically

## Troubleshooting

### Common Issues

#### MacAgent Won't Start

**Symptoms**: Application crashes on startup or shows error dialog

**Solutions**:
1. Check log files in `logs/system_YYYYMMDD.log`
2. Verify Python version with `python --version`
3. Reinstall dependencies with `pip install -r requirements.txt`
4. Clear configuration with `python -m MacAgent.app --reset-config`

#### Workflow Execution Fails

**Symptoms**: Workflow starts but fails to complete

**Solutions**:
1. Check the workflow log in `logs/workflows.log`
2. Verify all required permissions are granted
3. Run diagnostic check: `diagnostics run`
4. Try executing workflow in debug mode: `workflow run [name] --debug`

#### Permission Errors

**Symptoms**: Operations fail with "Permission denied" messages

**Solutions**:
1. Check current permissions with `permissions list`
2. Grant required permissions with `permissions grant [name]`
3. For temporary elevation: `permissions elevate [name] --duration 10`
4. If persistent, check system permissions in System Preferences

### Diagnostic Tools

MacAgent includes built-in diagnostic tools:

- `diagnostics report`: Generate a system report
- `diagnostics test`: Run system tests
- `diagnostics logs`: View diagnostic logs
- `diagnostics fix`: Attempt to fix common issues

### System Logs

Log files are stored in the `logs` directory:

- `system_YYYYMMDD.log`: General system logs
- `permissions_YYYYMMDD.log`: Permission-related events
- `safety_YYYYMMDD.log`: Safety protocol activities
- `security_YYYYMMDD.log`: Security-related events
- `diagnostics_YYYYMMDD.log`: Diagnostic information
- `workflows.log`: Workflow execution logs

## FAQ

### General Questions

**Q: Is MacAgent free to use?**  
A: MacAgent is available in both free and premium versions. The free version includes core functionality, while the premium version adds advanced automation and integrations.

**Q: Does MacAgent run in the background?**  
A: Yes, MacAgent can run in the background to monitor events and trigger workflows. This behavior can be configured in settings.

**Q: Does MacAgent send my data to the cloud?**  
A: No, MacAgent processes all data locally on your Mac. No user data is sent to external servers unless explicitly configured for specific integrations.

### Workflow Questions

**Q: Can I share workflows with other users?**  
A: Yes, you can export workflows and share them with other MacAgent users using the `workflow export` and `workflow import` commands.

**Q: How many steps can a workflow have?**  
A: There is no fixed limit on workflow steps, but complex workflows may impact performance. We recommend keeping workflows focused on specific tasks.

**Q: Can workflows interact with third-party applications?**  
A: Yes, MacAgent can interact with any application that provides AppleScript support or command-line interfaces.

### Security Questions

**Q: How does MacAgent protect my sensitive information?**  
A: MacAgent uses encryption for stored credentials and implements strict permission controls to limit access to sensitive resources.

**Q: Can MacAgent run potentially destructive operations?**  
A: MacAgent includes safety protocols that require confirmation for potentially destructive operations and provides simulation capabilities to preview effects.

## Support

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [online documentation](https://macagent.example.com/docs)
2. Join our [community forum](https://community.macagent.example.com)
3. Submit an issue on our [GitHub repository](https://github.com/yourusername/MacAgent/issues)

### Contributing

We welcome contributions to MacAgent! See our [developer guide](developer_guide.md) for information on how to contribute.

### Updates

MacAgent checks for updates automatically. To manually check for updates:

```
update check
```

To install available updates:

```
update install
```

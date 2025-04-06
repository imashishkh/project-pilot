# MacAgent Troubleshooting Guide

This guide provides solutions for common issues you might encounter while using MacAgent.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Startup Problems](#startup-problems)
3. [Permission Issues](#permission-issues)
4. [Workflow Errors](#workflow-errors)
5. [Performance Problems](#performance-problems)
6. [Security Concerns](#security-concerns)
7. [Diagnostic Tools](#diagnostic-tools)
8. [Log Files](#log-files)
9. [Command Line Troubleshooting](#command-line-troubleshooting)
10. [Contacting Support](#contacting-support)

## Installation Issues

### Failed Installation

**Issue**: Installation fails with an error message.

**Solutions**:
1. Verify your system meets the minimum requirements (macOS 12.0 or later)
2. Ensure you have sufficient disk space (at least 1GB free)
3. Try running the installer with administrator privileges:
   ```
   sudo ./install_macagent.sh
   ```
4. Check if any antivirus software is blocking the installation
5. Install any pending macOS updates and try again

### Missing Dependencies

**Issue**: After installation, MacAgent reports missing dependencies.

**Solutions**:
1. Run the dependency checker:
   ```
   macagent diagnose dependencies
   ```
2. Install missing dependencies manually:
   ```
   pip install -r requirements.txt
   ```
3. For system-level dependencies, use Homebrew:
   ```
   brew install <package-name>
   ```

### Upgrade Errors

**Issue**: Errors when upgrading from a previous version.

**Solutions**:
1. Backup your configuration:
   ```
   macagent backup config
   ```
2. Uninstall the previous version completely:
   ```
   macagent uninstall --keep-config
   ```
3. Install the new version
4. If problems persist, try a clean installation (without keeping the configuration)

## Startup Problems

### MacAgent Won't Start

**Issue**: Application doesn't start or crashes immediately after launch.

**Solutions**:
1. Check if the process is already running:
   ```
   ps aux | grep macagent
   ```
2. Kill any stale processes:
   ```
   pkill -f macagent
   ```
3. Check for corrupted configuration:
   ```
   macagent validate-config
   ```
4. Reset to default configuration:
   ```
   macagent reset-config
   ```
5. Check system logs for errors:
   ```
   log show --predicate 'processImagePath contains "macagent"' --last 30m
   ```

### Silent Crashes

**Issue**: MacAgent starts but silently disappears after a few seconds.

**Solutions**:
1. Launch in debug mode:
   ```
   macagent --debug
   ```
2. Check crash reports:
   ```
   open ~/Library/Logs/DiagnosticReports/
   ```
3. Verify file permissions:
   ```
   sudo chown -R $(whoami) ~/Library/Application\ Support/MacAgent/
   ```
4. Rebuild cache files:
   ```
   macagent rebuild-cache
   ```

### Menu Bar Icon Missing

**Issue**: MacAgent is running but the menu bar icon is not visible.

**Solutions**:
1. Check if MacAgent is running:
   ```
   pgrep -l macagent
   ```
2. Restart the UI component:
   ```
   macagent restart-ui
   ```
3. Check if the icon is hidden in the menu bar overflow area
4. Reset macOS menu bar:
   ```
   killall SystemUIServer
   ```

## Permission Issues

### Feature Requires Permissions

**Issue**: MacAgent shows "This feature requires additional permissions" message.

**Solutions**:
1. Open the permissions panel:
   ```
   macagent permissions
   ```
2. Grant the specific permission needed
3. For system-level permissions, check System Preferences > Security & Privacy
4. If a permission was previously denied, you may need to reset it:
   ```
   tccutil reset All com.macagent.app
   ```

### Access Denied Errors

**Issue**: "Access Denied" error when MacAgent tries to perform an operation.

**Solutions**:
1. Check current permission status:
   ```
   macagent permissions status
   ```
2. Grant temporary elevated permissions:
   ```
   macagent permissions elevate --duration 5m
   ```
3. For file access issues, check file permissions:
   ```
   ls -la /path/to/file
   ```
4. Add MacAgent to Full Disk Access in System Preferences if necessary

### Automation Permission Prompts

**Issue**: Constant permission prompts when controlling other applications.

**Solutions**:
1. Grant automation permissions in System Preferences > Security & Privacy > Privacy > Automation
2. Pre-approve applications:
   ```
   macagent approve-apps --list "Safari,Mail,Calendar"
   ```
3. Create an automation profile to avoid repeated prompts:
   ```
   macagent create-profile standard
   ```

## Workflow Errors

### Workflow Won't Run

**Issue**: A workflow doesn't execute when triggered.

**Solutions**:
1. Check workflow status:
   ```
   macagent workflow status "Workflow Name"
   ```
2. Verify the workflow is enabled:
   ```
   macagent workflow enable "Workflow Name"
   ```
3. Test running the workflow manually:
   ```
   macagent workflow run "Workflow Name" --debug
   ```
4. Check for missing dependencies the workflow might need
5. Verify that the necessary applications are installed

### Steps Fail in Workflow

**Issue**: Specific steps in a workflow fail to execute properly.

**Solutions**:
1. Run the workflow in debug mode:
   ```
   macagent workflow run "Workflow Name" --debug
   ```
2. Check the logs for that specific workflow:
   ```
   macagent logs --workflow "Workflow Name"
   ```
3. Test the failing step individually:
   ```
   macagent workflow run-step "Workflow Name" --step 3
   ```
4. Check if the application being controlled is responsive
5. Update the step parameters if they contain outdated information

### Scheduled Workflows Don't Run

**Issue**: Workflows scheduled to run automatically never execute.

**Solutions**:
1. Verify MacAgent is running in the background:
   ```
   macagent status
   ```
2. Check scheduled tasks:
   ```
   macagent schedule list
   ```
3. Ensure your Mac isn't asleep when workflows are scheduled
4. Check if MacAgent has permission to run in the background
5. Verify the date and time on your Mac are correct

### Workflows Running Too Slowly

**Issue**: Workflows take too long to complete.

**Solutions**:
1. Check workflow execution time:
   ```
   macagent workflow stats "Workflow Name"
   ```
2. Optimize wait times between steps:
   ```
   macagent workflow optimize "Workflow Name"
   ```
3. Reduce the scope of search operations
4. Break complex workflows into smaller, more focused ones
5. Make sure applications being controlled are not in a busy state

## Performance Problems

### High CPU Usage

**Issue**: MacAgent is using excessive CPU resources.

**Solutions**:
1. Check which component is using CPU:
   ```
   macagent diagnose performance
   ```
2. Stop unnecessary background tasks:
   ```
   macagent tasks stop --non-critical
   ```
3. Reduce logging level:
   ```
   macagent configure logging --level warning
   ```
4. Limit the number of concurrent workflows:
   ```
   macagent configure workflows --max-concurrent 2
   ```
5. Update to the latest version which may contain performance improvements

### Memory Leaks

**Issue**: MacAgent's memory usage grows continuously over time.

**Solutions**:
1. Monitor memory usage:
   ```
   macagent diagnose memory --watch
   ```
2. Restart the application to free memory:
   ```
   macagent restart
   ```
3. Check for problematic workflows:
   ```
   macagent workflow analyze-memory
   ```
4. Disable memory-intensive features:
   ```
   macagent configure memory --optimize-for low
   ```
5. Schedule regular restarts:
   ```
   macagent schedule restart --daily
   ```

### Slow Startup

**Issue**: MacAgent takes a long time to start.

**Solutions**:
1. Reduce the number of workflows loaded at startup:
   ```
   macagent configure startup --minimal
   ```
2. Clean up cached data:
   ```
   macagent cleanup cache
   ```
3. Disable non-essential plugins:
   ```
   macagent plugins disable --non-essential
   ```
4. Check for disk issues that might slow down startup:
   ```
   diskutil verifyVolume /
   ```
5. Rebuild the configuration index:
   ```
   macagent rebuild-index
   ```

## Security Concerns

### Security Alert Messages

**Issue**: MacAgent displays security alert messages.

**Solutions**:
1. Check security log:
   ```
   macagent security log
   ```
2. Scan for potential threats:
   ```
   macagent security scan
   ```
3. Update your security configuration:
   ```
   macagent security configure
   ```
4. Verify if the alert is a false positive:
   ```
   macagent security verify-alert "Alert ID"
   ```
5. Report false positives to improve detection:
   ```
   macagent security report-false-positive "Alert ID"
   ```

### Unauthorized Access Attempts

**Issue**: MacAgent detects unauthorized access attempts.

**Solutions**:
1. Change your MacAgent credentials:
   ```
   macagent credentials reset
   ```
2. Check recent access logs:
   ```
   macagent security access-log
   ```
3. Enable two-factor authentication:
   ```
   macagent security enable-2fa
   ```
4. Lock down permissions:
   ```
   macagent permissions lockdown
   ```
5. Revoke all existing sessions:
   ```
   macagent security revoke-sessions
   ```

### Data Privacy Concerns

**Issue**: Concerns about what data MacAgent accesses or stores.

**Solutions**:
1. View data access report:
   ```
   macagent privacy report
   ```
2. Configure privacy settings:
   ```
   macagent privacy configure
   ```
3. Enable private mode:
   ```
   macagent privacy enable-private-mode
   ```
4. Delete stored data:
   ```
   macagent privacy delete-data
   ```
5. Audit third-party plugins:
   ```
   macagent plugins audit-privacy
   ```

## Diagnostic Tools

### Built-in Diagnostics

MacAgent includes several built-in diagnostic tools:

1. Run a complete system test:
   ```
   macagent diagnose system
   ```
2. Check network connectivity:
   ```
   macagent diagnose network
   ```
3. Verify file system access:
   ```
   macagent diagnose filesystem
   ```
4. Test application control capabilities:
   ```
   macagent diagnose app-control
   ```
5. Generate a comprehensive diagnostic report:
   ```
   macagent diagnose report --full
   ```

### Fixing Common Problems

Use the automatic repair function to fix common issues:

1. Fix configuration issues:
   ```
   macagent repair config
   ```
2. Repair database issues:
   ```
   macagent repair database
   ```
3. Resolve permission problems:
   ```
   macagent repair permissions
   ```
4. Fix workflow issues:
   ```
   macagent repair workflows
   ```
5. Comprehensive repair:
   ```
   macagent repair all
   ```

### Component Testing

Test specific components individually:

1. Test the permission manager:
   ```
   macagent test-component permission_manager
   ```
2. Test the workflow automator:
   ```
   macagent test-component workflow_automator
   ```
3. Test the security monitor:
   ```
   macagent test-component security_monitor
   ```
4. Test the diagnostic system itself:
   ```
   macagent test-component diagnostic_system
   ```
5. Test all components:
   ```
   macagent test-components
   ```

## Log Files

### Accessing Log Files

MacAgent keeps several log files you can examine:

1. Main application log:
   ```
   open ~/Library/Logs/MacAgent/macagent.log
   ```
2. Workflow execution logs:
   ```
   open ~/Library/Logs/MacAgent/workflows/
   ```
3. Security logs:
   ```
   open ~/Library/Logs/MacAgent/security.log
   ```
4. System diagnostics logs:
   ```
   open ~/Library/Logs/MacAgent/diagnostics.log
   ```
5. Performance monitoring logs:
   ```
   open ~/Library/Logs/MacAgent/performance.log
   ```

### Log Management

Tools for managing MacAgent logs:

1. Clear old logs:
   ```
   macagent logs clear --older-than 30d
   ```
2. Archive logs:
   ```
   macagent logs archive
   ```
3. Change logging verbosity:
   ```
   macagent configure logging --level debug
   ```
4. Enable component-specific logging:
   ```
   macagent configure logging --component workflow_automator --level debug
   ```
5. Export logs for support:
   ```
   macagent logs export --support-bundle
   ```

### Log Analysis

Tools to help you analyze logs:

1. Search logs for errors:
   ```
   macagent logs grep "error"
   ```
2. View workflow execution timeline:
   ```
   macagent logs timeline "Workflow Name"
   ```
3. Generate a log summary:
   ```
   macagent logs summarize
   ```
4. Compare logs before and after a change:
   ```
   macagent logs diff before.log after.log
   ```
5. Visualize log patterns:
   ```
   macagent logs visualize --type errors
   ```

## Command Line Troubleshooting

### Command Syntax Issues

**Issue**: Command line commands return syntax errors.

**Solutions**:
1. Check command syntax:
   ```
   macagent help <command>
   ```
2. Use command completion:
   ```
   macagent completion install
   ```
3. See examples for a specific command:
   ```
   macagent examples <command>
   ```
4. Try interactive mode:
   ```
   macagent interactive
   ```
5. Enable verbose output to see what's happening:
   ```
   macagent <command> --verbose
   ```

### Authentication Issues

**Issue**: Command line operations fail with authentication errors.

**Solutions**:
1. Verify your authentication status:
   ```
   macagent auth status
   ```
2. Re-authenticate:
   ```
   macagent auth login
   ```
3. Reset credentials if necessary:
   ```
   macagent auth reset
   ```
4. Check permissions for the command:
   ```
   macagent permissions check <command>
   ```
5. Run with elevated privileges (when appropriate):
   ```
   sudo macagent <command>
   ```

### API Access Problems

**Issue**: Scripts or custom tools can't access the MacAgent API.

**Solutions**:
1. Check API status:
   ```
   macagent api status
   ```
2. Generate a new API key:
   ```
   macagent api generate-key
   ```
3. Test API connectivity:
   ```
   macagent api test
   ```
4. Enable API access:
   ```
   macagent api enable
   ```
5. Check API documentation:
   ```
   macagent api docs
   ```

## Contacting Support

If you're unable to resolve your issue using this guide, please contact support:

1. Generate a support bundle:
   ```
   macagent support-bundle create
   ```
2. Contact the support team with your bundle:
   - Email: support@macagent.example.com
   - Support portal: https://support.macagent.example.com
   - Community forums: https://community.macagent.example.com

3. When contacting support, please include:
   - Your MacAgent version
   - macOS version
   - A detailed description of the issue
   - Steps to reproduce the problem
   - Any error messages you're seeing
   - The support bundle generated above 
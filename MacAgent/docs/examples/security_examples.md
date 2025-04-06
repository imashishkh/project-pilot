# MacAgent Security Examples

This document provides practical examples of how to use MacAgent's security features to protect your system while maintaining productivity.

## Table of Contents

1. [Introduction](#introduction)
2. [Permission Management Examples](#permission-management-examples)
   - [Basic Permission Controls](#basic-permission-controls)
   - [Advanced Permission Configurations](#advanced-permission-configurations)
   - [Temporary Permission Elevation](#temporary-permission-elevation)
3. [Safety Protocol Examples](#safety-protocol-examples)
   - [Safe File Operations](#safe-file-operations)
   - [Safe System Modifications](#safe-system-modifications)
   - [Operation Simulation](#operation-simulation)
4. [Security Monitoring Examples](#security-monitoring-examples)
   - [Credential Management](#credential-management)
   - [Data Protection](#data-protection)
   - [Security Auditing](#security-auditing)
5. [Integration Examples](#integration-examples)
   - [Secure Workflow Design](#secure-workflow-design)
   - [Cross-Component Security](#cross-component-security)
6. [Best Practices](#best-practices)

## Introduction

MacAgent includes comprehensive security features designed to protect your system while allowing automation to work efficiently. This guide provides practical examples of how to use these security features in real-world scenarios.

The three main security components are:

1. **Permission Manager**: Controls what MacAgent can access
2. **Safety Protocol**: Ensures potentially destructive operations are handled safely
3. **Security Monitor**: Protects sensitive information and detects security risks

## Permission Management Examples

### Basic Permission Controls

#### View Current Permissions

To view the current permissions MacAgent has:

```
macagent permissions list
```

Example output:
```
FileSystem:
  Read: ~/Documents [GRANTED]
  Write: ~/Documents/MacAgent [GRANTED]
  Read: /Applications [GRANTED]
Applications:
  Control: Safari [GRANTED]
  Control: Mail [PROMPT]
  Control: Calendar [DENIED]
System:
  Network: Outbound [GRANTED]
  Notifications: [GRANTED]
```

#### Request a New Permission

To request permission to read files in a specific location:

```
macagent permissions request file_read --path "~/Projects" --reason "Need to access project files"
```

This will prompt you with details about the permission being requested:

```
Permission Request:
- Type: file_read
- Path: /Users/username/Projects
- Reason: Need to access project files

Grant this permission? [Y/n/details]: 
```

Type `details` for more information or `Y` to grant.

#### Revoke a Permission

To revoke a previously granted permission:

```
macagent permissions revoke file_write --path "~/Documents/Sensitive"
```

### Advanced Permission Configurations

#### Create a Permission Profile

Permission profiles help manage sets of permissions for specific tasks:

```
macagent permissions create-profile "project-automation" \
  --allow file_read:~/Projects \
  --allow file_write:~/Projects/Output \
  --allow app_control:Terminal \
  --deny file_read:~/Documents/Personal
```

#### Apply a Permission Profile

To apply a saved permission profile:

```
macagent permissions apply-profile "project-automation"
```

#### Create Granular File Permissions

For very specific access controls:

```
macagent permissions set file_read \
  --allow "~/Projects/*.md" \
  --allow "~/Projects/*/config.json" \
  --deny "~/Projects/*/credentials.json"
```

### Temporary Permission Elevation

#### Elevate for a Single Operation

For operations that require temporary elevated permissions:

```
macagent permissions elevate file_write --path "/Applications/MacAgent" --duration 5m --reason "Installing update"
```

This will grant the permission for 5 minutes then automatically revoke it.

#### Scheduled Permission Elevation

Schedule permission elevation for planned maintenance:

```
macagent permissions schedule-elevation system_settings \
  --start "2023-07-15T01:00:00" \
  --duration 30m \
  --reason "Scheduled system maintenance"
```

#### Permission Auditing

Track how permissions are being used:

```
macagent permissions audit --last 7d
```

Example output:
```
Permission: file_read:~/Documents
- Used: 32 times
- Last used: 2023-07-10 15:42
- Workflows: "Document Backup", "Meeting Prep"

Permission: network:outbound
- Used: 128 times
- Last used: 2023-07-12 09:15
- Workflows: "Email Organization", "News Summary"
```

## Safety Protocol Examples

### Safe File Operations

#### Safe File Deletion

To safely delete files with the ability to undo:

```
macagent safety delete ~/Downloads/temp-files/*.tmp
```

If something goes wrong, you can recover the files:

```
macagent safety undo operation-12345
```

#### Safe Batch Operations

For batch operations with added safety:

```
macagent safety batch-move ~/Downloads/*.pdf ~/Documents/PDFs \
  --simulate-first \
  --confirm-each 5 \
  --create-backup
```

This will:
1. First simulate the operation
2. Ask for confirmation after every 5 files
3. Create backups before moving

#### Safe File Content Replacement

To safely modify file contents:

```
macagent safety replace-in-files "old API key" "new API key" ~/Projects/config/*.json \
  --backup \
  --show-preview
```

### Safe System Modifications

#### Safe System Setting Changes

Change system settings with safety guards:

```
macagent safety change-setting network.proxy.enabled true \
  --expire 2h \
  --restore-after \
  --notify-before-restore
```

This will:
1. Change the setting
2. Automatically restore it after 2 hours
3. Send a notification before restoring

#### Safe Application Control

Control applications with safety mechanisms:

```
macagent safety control-app "Mail" clear-cache \
  --confirm-action \
  --prevent-data-loss
```

#### Safe Script Execution

Execute scripts with safety controls:

```
macagent safety run-script ~/Scripts/cleanup.sh \
  --sandbox \
  --resource-limits "cpu:50%,memory:1GB" \
  --timeout 5m
```

### Operation Simulation

#### Simulate File Operations

Preview what would happen without executing:

```
macagent safety simulate organize-downloads \
  --rules "move:*.pdf:~/Documents/PDFs,move:*.jpg:~/Pictures"
```

Example output:
```
Simulation Results:
Move 23 PDF files to ~/Documents/PDFs
  - ~/Downloads/report.pdf
  - ~/Downloads/invoice.pdf
  - ...
Move 15 JPG files to ~/Pictures
  - ~/Downloads/photo1.jpg
  - ~/Downloads/screenshot.jpg
  - ...
```

#### Simulate System Impact

Assess the impact of operations on system resources:

```
macagent safety impact-analysis large-backup \
  --source ~/Projects \
  --destination /Volumes/Backup
```

Example output:
```
Impact Analysis:
- Storage required: 2.3 GB
- Estimated time: 4-5 minutes
- Network usage: None (local operation)
- CPU load: Medium (30-40%)
- Risk level: Low
```

#### Simulated Dry Run

Perform a full dry run of a workflow:

```
macagent safety dry-run "Weekly Cleanup" \
  --detailed-output \
  --show-risks
```

## Security Monitoring Examples

### Credential Management

#### Store Credentials Securely

Store credentials in the secure credential storage:

```
macagent security store-credential "aws-api-key" \
  --value "AKIAIOSFODNN7EXAMPLE" \
  --description "AWS API Key for backups" \
  --expires "2023-12-31"
```

#### Retrieve Credentials

Retrieve stored credentials for use in workflows:

```
macagent security get-credential "aws-api-key" \
  --use-in "Cloud Backup" \
  --log-access
```

#### Rotate Credentials

Automatically rotate credentials:

```
macagent security rotate-credential "database-password" \
  --generator "strong-password" \
  --length 24 \
  --notify-services "db-server,app-server"
```

### Data Protection

#### Detect Sensitive Information

Scan files for sensitive information:

```
macagent security scan-pii ~/Documents/exports/ \
  --types "credit-card,ssn,phone,email" \
  --detailed-report
```

Example output:
```
PII Scan Results:
- Found 3 credit card numbers in exports_2023.csv
- Found 12 email addresses in contacts.xlsx
- Found 2 SSNs in employee_data.docx
```

#### Redact Sensitive Information

Automatically redact sensitive information:

```
macagent security redact ~/Documents/exports/customer_data.csv \
  --output ~/Documents/exports/customer_data_redacted.csv \
  --types "credit-card,ssn" \
  --keep-format
```

#### Encrypt Sensitive Files

Encrypt files containing sensitive information:

```
macagent security encrypt ~/Documents/tax_documents/ \
  --output ~/Documents/encrypted/ \
  --password-prompt \
  --delete-originals
```

### Security Auditing

#### Generate Security Report

Create a comprehensive security audit report:

```
macagent security audit-report \
  --period "last-month" \
  --include "permissions,access,operations" \
  --format pdf \
  --output ~/Documents/security_audit.pdf
```

#### Monitor Unusual Activity

Set up monitoring for unusual activity:

```
macagent security monitor unusual-activity \
  --baseline "2-weeks" \
  --notify-on "high-severity" \
  --ignore-patterns "routine-backup"
```

#### Security Alerts Configuration

Configure security alerts:

```
macagent security configure-alerts \
  --level warning \
  --channels "notification,email" \
  --email "admin@example.com" \
  --quiet-hours "22:00-08:00"
```

## Integration Examples

### Secure Workflow Design

#### Create a Security-First Workflow

Design a workflow with security as the priority:

```
macagent workflow create "Secure Document Processing" \
  --security-level high \
  --permission-profile minimal \
  --require-explicit-approval
```

Add secure steps that follow least-privilege principles:

```
macagent workflow add-step "Finder" "find_files" \
  --parameters '{"location":"~/Documents/Input", "pattern":"*.pdf"}' \
  --permission "file_read:~/Documents/Input"

macagent workflow add-step "Security" "scan_files" \
  --parameters '{"scan_type":"malware"}' \
  --fail-workflow-if-threats-found

macagent workflow add-step "Finder" "process_files" \
  --parameters '{"action":"convert_to_text"}' \
  --temporary-output "~/tmp/processing"

macagent workflow add-step "Security" "redact_pii" \
  --parameters '{"types":["email","phone","address"]}' \
  --audit-trail

macagent workflow add-step "Finder" "move_files" \
  --parameters '{"destination":"~/Documents/Processed"}' \
  --permission "file_write:~/Documents/Processed"

macagent workflow add-step "Security" "clean_temp" \
  --parameters '{"secure_delete":true}' \
  --run-even-if-workflow-fails
```

#### Auto-Apply Security Policies

Automatically apply security policies to workflows:

```
macagent security policy apply "data-protection" \
  --workflows "Document Processing,Email Export" \
  --enforce \
  --remediate-issues
```

### Cross-Component Security

#### Integrate Permission Manager with Workflows

Configure how workflows interact with permissions:

```
macagent integrate permissions workflows \
  --auto-request \
  --detailed-reasons \
  --remember-grants
```

#### Connect Safety Protocols to Monitoring

Link safety protocols with security monitoring:

```
macagent integrate safety monitoring \
  --alert-on-risky-operations \
  --record-operations \
  --threshold medium
```

#### Security Dashboard Setup

Create a security dashboard for system-wide visibility:

```
macagent security dashboard \
  --components "all" \
  --refresh-interval 5m \
  --highlight-issues \
  --show-history 7d
```

## Best Practices

Here are some best practices for using MacAgent's security features effectively:

1. **Start Restrictive and Open Up**: Begin with minimal permissions and add only what's needed.

   ```
   macagent permissions set-default --deny-all
   macagent permissions request specific_permission --as-needed
   ```

2. **Use Time-Limited Permissions**: For sensitive operations, use time-limited permissions.

   ```
   macagent permissions elevate sensitive_operation --duration 5m --auto-revoke
   ```

3. **Create Workflow-Specific Profiles**: Define permission profiles for specific workflows.

   ```
   macagent permissions create-profile "workflow-name-profile" --based-on-workflow "Workflow Name"
   ```

4. **Regularly Audit Permissions**: Review and clean up permissions regularly.

   ```
   macagent permissions audit --unused-since 30d --auto-revoke
   ```

5. **Test in Safe Mode**: Test workflows in a restricted environment first.

   ```
   macagent workflow test-run "Workflow Name" --safe-mode --simulated-environment
   ```

6. **Use Operation Simulation**: Always simulate destructive operations first.

   ```
   macagent safety simulate operation --before-execution --detailed-preview
   ```

7. **Implement Defense in Depth**: Use multiple security components together.

   ```
   macagent security defense-in-depth --enable --components "all"
   ```

8. **Set Up Security Notifications**: Stay informed about security events.

   ```
   macagent security notifications configure --levels "warning,critical" --immediate
   ```

9. **Regularly Update Security Policies**: Keep security configurations current.

   ```
   macagent security policies update --from-templates --adapt-to-usage
   ```

10. **Document Security Decisions**: Keep records of security configuration decisions.

    ```
    macagent security document-decision "Limited network access" --reason "Only needed for API calls" --approved-by "Admin"
    ``` 
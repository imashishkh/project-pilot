# MacAgent Workflow Examples

This document provides step-by-step examples of how to use MacAgent's workflow automation capabilities for common tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Workflow Concepts](#basic-workflow-concepts)
3. [File Management Examples](#file-management-examples)
   - [Daily Document Backup](#daily-document-backup)
   - [Organizing Downloads Folder](#organizing-downloads-folder)
4. [Email Management Examples](#email-management-examples)
   - [Email Filtering and Organization](#email-filtering-and-organization)
   - [Automatic Email Responses](#automatic-email-responses)
5. [Productivity Examples](#productivity-examples)
   - [Meeting Preparation Workflow](#meeting-preparation-workflow)
   - [Research Collection Workflow](#research-collection-workflow)
6. [System Maintenance Examples](#system-maintenance-examples)
   - [Weekly Cleanup Routine](#weekly-cleanup-routine)
   - [Software Update Workflow](#software-update-workflow)
7. [Cross-Application Workflows](#cross-application-workflows)
   - [Web Research to Document Workflow](#web-research-to-document-workflow)
   - [Project Management Workflow](#project-management-workflow)
8. [Advanced Workflow Techniques](#advanced-workflow-techniques)
   - [Conditional Branching](#conditional-branching)
   - [Error Handling](#error-handling)
   - [Using Environment Variables](#using-environment-variables)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Introduction

MacAgent workflows allow you to automate repetitive tasks across multiple applications. A workflow consists of a series of steps that are executed in sequence, with each step performing a specific action in a specific application.

This guide provides concrete examples of useful workflows, with step-by-step instructions on how to create, customize, and run them.

## Basic Workflow Concepts

Before diving into specific examples, here are the basic concepts of MacAgent workflows:

- **Workflow**: A sequence of automated steps that accomplish a task
- **Step**: A single action within a workflow
- **Action**: What a step does (e.g., find files, send email)
- **Parameters**: Configuration options for an action
- **Wait Time**: Optional delay between steps

### Creating a Simple Workflow

To create a workflow using the command line:

```
workflow create "My Workflow"
```

To create a workflow using the GUI:
1. Open MacAgent
2. Click "Workflows" in the sidebar
3. Click "New Workflow"
4. Enter a name for your workflow

### Recording vs. Building

You can create workflows in two ways:
1. **Recording**: MacAgent observes and records your actions
2. **Building**: You manually add steps to a workflow

This guide will show examples of both approaches.

## File Management Examples

### Daily Document Backup

This workflow automatically backs up your important documents to an external drive or cloud storage.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Daily Document Backup"
   ```

2. Add steps to find recently modified documents:
   ```
   workflow add-step "Finder" "find_files" --parameters '{"location":"~/Documents", "modified_since":"1d"}'
   ```

3. Add a step to compress the files:
   ```
   workflow add-step "Finder" "compress_files" --parameters '{"output_path":"~/Backups/daily_backup_$(date +%Y-%m-%d).zip"}'
   ```

4. Add a step to copy to an external location (optional):
   ```
   workflow add-step "Finder" "copy_files" --parameters '{"source":"~/Backups/daily_backup_$(date +%Y-%m-%d).zip", "destination":"/Volumes/Backup/Documents/"}'
   ```

5. Schedule the workflow to run daily:
   ```
   workflow schedule "Daily Document Backup" --time "20:00" --repeat "daily"
   ```

#### Example of Running Manually:

```
workflow run "Daily Document Backup"
```

### Organizing Downloads Folder

This workflow automatically organizes files in your Downloads folder based on their type.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Organize Downloads"
   ```

2. Find all files in Downloads older than 1 day:
   ```
   workflow add-step "Finder" "find_files" --parameters '{"location":"~/Downloads", "older_than":"1d"}'
   ```

3. Move image files to Pictures:
   ```
   workflow add-step "Finder" "move_files_by_type" --parameters '{"file_types":["jpg","png","gif","tiff","jpeg"], "destination":"~/Pictures/Downloads"}'
   ```

4. Move document files to Documents:
   ```
   workflow add-step "Finder" "move_files_by_type" --parameters '{"file_types":["pdf","doc","docx","xls","xlsx","ppt","pptx"], "destination":"~/Documents/Downloads"}'
   ```

5. Move video files to Movies:
   ```
   workflow add-step "Finder" "move_files_by_type" --parameters '{"file_types":["mp4","mov","avi","mkv"], "destination":"~/Movies/Downloads"}'
   ```

6. Schedule to run weekly:
   ```
   workflow schedule "Organize Downloads" --time "18:00" --repeat "weekly" --day "Friday"
   ```

## Email Management Examples

### Email Filtering and Organization

This workflow helps you manage your inbox by automatically sorting emails into folders.

#### Steps to Create (Recording Method):

1. Start recording a new workflow:
   ```
   workflow record "Email Organization"
   ```

2. Open Mail app
3. Create a rule to move emails from important contacts to a "Priority" folder
4. Create a rule to move newsletter emails to a "Newsletters" folder
5. Create a rule to flag emails containing specific keywords
6. Stop recording:
   ```
   workflow stop-recording
   ```

7. Schedule to run every hour:
   ```
   workflow schedule "Email Organization" --repeat "hourly"
   ```

### Automatic Email Responses

This workflow sends automatic responses to emails matching certain criteria when you're on vacation.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Vacation Auto-Response"
   ```

2. Add a step to check for new emails:
   ```
   workflow add-step "Mail" "check_new_emails" --parameters '{"account":"Personal"}'
   ```

3. Add a step to filter work-related emails:
   ```
   workflow add-step "Mail" "filter_emails" --parameters '{"criteria":{"from_domain":"company.com"}}'
   ```

4. Add a step to send an auto-response:
   ```
   workflow add-step "Mail" "send_response" --parameters '{"template":"vacation_response", "include_original":true}'
   ```

5. Enable the workflow when going on vacation:
   ```
   workflow enable "Vacation Auto-Response" --until "2023-07-15"
   ```

## Productivity Examples

### Meeting Preparation Workflow

This workflow helps you prepare for upcoming meetings by gathering relevant information.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Meeting Prep"
   ```

2. Add a step to check calendar for upcoming meetings:
   ```
   workflow add-step "Calendar" "find_next_events" --parameters '{"hours_ahead":1, "title_contains":"Meeting"}'
   ```

3. Add a step to gather recent emails from meeting participants:
   ```
   workflow add-step "Mail" "find_emails" --parameters '{"from":"$(meeting_attendees)", "days_back":7}'
   ```

4. Add a step to create a meeting notes document:
   ```
   workflow add-step "TextEdit" "create_document" --parameters '{"template":"meeting_notes", "title":"Meeting Notes - $(meeting_title)", "save_path":"~/Documents/Meeting Notes/"}'
   ```

5. Add a step to open relevant documents:
   ```
   workflow add-step "Finder" "find_files" --parameters '{"location":"~/Documents", "content_contains":"$(meeting_topic)"}'
   ```

6. Schedule to run 15 minutes before meetings:
   ```
   workflow schedule "Meeting Prep" --before-calendar-events "Meeting" --minutes 15
   ```

### Research Collection Workflow

This workflow helps collect and organize research materials on a specific topic.

#### Steps to Create (Recording Method):

1. Start recording a new workflow:
   ```
   workflow record "Research Collection"
   ```

2. Open Safari and search for your research topic
3. Save relevant articles to Reading List
4. Export Reading List to a folder
5. Create a summary document with links
6. Stop recording:
   ```
   workflow stop-recording
   ```

7. Customize the workflow with variables:
   ```
   workflow edit "Research Collection" --replace "search_term" --with "$(research_topic)"
   ```

## System Maintenance Examples

### Weekly Cleanup Routine

This workflow performs routine system maintenance tasks to keep your Mac running smoothly.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Weekly Cleanup"
   ```

2. Add a step to empty the trash:
   ```
   workflow add-step "Finder" "empty_trash" --parameters '{"confirm":false}'
   ```

3. Add a step to clear browser caches:
   ```
   workflow add-step "Safari" "clear_cache" --parameters '{"keep_logins":true}'
   ```

4. Add a step to remove temporary files:
   ```
   workflow add-step "System" "remove_temp_files" --parameters '{"older_than":"7d"}'
   ```

5. Add a step to run First Aid on the disk:
   ```
   workflow add-step "Disk Utility" "first_aid" --parameters '{"volume":"Macintosh HD"}'
   ```

6. Schedule to run weekly on Sunday:
   ```
   workflow schedule "Weekly Cleanup" --time "01:00" --repeat "weekly" --day "Sunday"
   ```

### Software Update Workflow

This workflow checks for and installs software updates during off-hours.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Software Update"
   ```

2. Add a step to check for system updates:
   ```
   workflow add-step "System" "check_updates" --parameters '{"include_apps":true}'
   ```

3. Add a conditional step to install non-critical updates:
   ```
   workflow add-step "System" "install_updates" --parameters '{"critical_only":false, "restart":false}'
   ```

4. Add notification on completion:
   ```
   workflow add-step "Notifications" "send_notification" --parameters '{"title":"Software Update Complete", "message":"Your system has been updated."}'
   ```

5. Schedule to run weekly during night hours:
   ```
   workflow schedule "Software Update" --time "03:00" --repeat "weekly" --day "Wednesday"
   ```

## Cross-Application Workflows

### Web Research to Document Workflow

This workflow collects information from various websites and compiles it into a document.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "Web Research Compiler"
   ```

2. Add a step to open Safari and search:
   ```
   workflow add-step "Safari" "search" --parameters '{"query":"$(research_topic)"}'
   ```

3. Add a step to extract content from multiple pages:
   ```
   workflow add-step "Safari" "extract_content" --parameters '{"selector":"article", "max_pages":5}'
   ```

4. Add a step to create a new document:
   ```
   workflow add-step "Pages" "create_document" --parameters '{"template":"research", "title":"Research on $(research_topic)"}'
   ```

5. Add a step to insert compiled content:
   ```
   workflow add-step "Pages" "insert_content" --parameters '{"content":"$(extracted_content)", "format":"markdown"}'
   ```

6. Add a step to save the document:
   ```
   workflow add-step "Pages" "save_document" --parameters '{"path":"~/Documents/Research/$(research_topic).pages"}'
   ```

### Project Management Workflow

This workflow helps set up a new project by creating necessary folders, documents, and setting up task tracking.

#### Steps to Create:

1. Create a new workflow:
   ```
   workflow create "New Project Setup"
   ```

2. Add a step to create project folders:
   ```
   workflow add-step "Finder" "create_folders" --parameters '{"base_path":"~/Projects/$(project_name)", "folders":["docs", "src", "resources", "meetings"]}'
   ```

3. Add a step to create project documents:
   ```
   workflow add-step "Pages" "create_document" --parameters '{"template":"project_plan", "title":"$(project_name) Plan", "save_path":"~/Projects/$(project_name)/docs/project_plan.pages"}'
   ```

4. Add a step to create a task list in Reminders:
   ```
   workflow add-step "Reminders" "create_list" --parameters '{"name":"Project: $(project_name)"}'
   ```

5. Add a step to add initial tasks:
   ```
   workflow add-step "Reminders" "add_items" --parameters '{"list":"Project: $(project_name)", "items":["Initial research", "Create timeline", "Set up team meeting"]}'
   ```

6. Add a step to create a calendar event for kickoff:
   ```
   workflow add-step "Calendar" "create_event" --parameters '{"title":"$(project_name) Kickoff", "date":"$(kickoff_date)", "duration":60, "invitees":"$(team_members)"}'
   ```

## Advanced Workflow Techniques

### Conditional Branching

This example shows how to add conditional logic to workflows:

```
workflow add-step "System" "check_condition" --parameters '{"condition":"$(free_space) < 10", "if_true":"Cleanup", "if_false":"Skip"}'
```

Where "Cleanup" and "Skip" are labels for different sections of your workflow.

### Error Handling

Add error handling to your workflows:

```
workflow add-step "System" "try_catch" --parameters '{"try":"RiskyOperation", "catch":"ErrorHandler", "finally":"Cleanup"}'
```

Where "RiskyOperation", "ErrorHandler", and "Cleanup" are labels for different sections of your workflow.

### Using Environment Variables

Use environment variables in your workflows:

```
workflow add-step "System" "set_variable" --parameters '{"name":"TODAY", "value":"$(date +%Y-%m-%d)"}'
```

Then reference this variable in subsequent steps:

```
workflow add-step "Finder" "create_folder" --parameters '{"path":"~/Backups/$(env.TODAY)"}'
```

## Troubleshooting Common Issues

### Workflow Not Running

If your workflow isn't running as scheduled:

1. Check if MacAgent is running in the background
2. Verify the workflow is enabled: `workflow status "Workflow Name"`
3. Check for permission issues: `workflow diagnose "Workflow Name"`

### Steps Failing

If specific steps in your workflow are failing:

1. Run the workflow in debug mode: `workflow run "Workflow Name" --debug`
2. Check the logs: `workflow logs "Workflow Name"`
3. Try running the failing step individually: `workflow run-step "Workflow Name" --step 3`

### Performance Issues

If your workflow is running slowly:

1. Reduce wait times between steps
2. Optimize file operations to handle fewer files
3. Use more specific criteria in search operations
4. Consider breaking large workflows into smaller ones 
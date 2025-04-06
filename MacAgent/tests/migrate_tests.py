#!/usr/bin/env python3
"""
Script to help with migrating tests to the new structure.

This script will:
1. Identify tests in old locations
2. Suggest appropriate locations in the new structure
3. Help copy tests to their new locations
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def find_test_files(directory):
    """Find all test files in a directory recursively."""
    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    return test_files


def categorize_test_file(file_path):
    """Categorize a test file into vision, interaction, intelligence, or core."""
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Determine the category based on imports and content
    if any(x in content for x in ["vision", "screen_capture", "element_detector", "image_processor"]):
        return "vision"
    elif any(x in content for x in ["interaction", "mouse", "keyboard", "applescript"]):
        return "interaction"
    elif any(x in content for x in ["intelligence", "llm", "planning", "decision"]):
        return "intelligence"
    else:
        return "core"


def determine_new_location(file_path, old_root, new_root):
    """Determine the new location for a test file."""
    # Get the relative path
    rel_path = os.path.relpath(file_path, old_root)
    
    # Determine the category
    category = categorize_test_file(file_path)
    
    # Determine whether it's a unit or integration test
    # Simple heuristic: if it imports from multiple modules, it's an integration test
    with open(file_path, 'r') as f:
        content = f.read()
    
    categories = ["vision", "interaction", "intelligence", "core"]
    imported_categories = sum(1 for cat in categories if cat in content)
    
    if imported_categories > 1:
        test_type = "integration"
    else:
        test_type = "unit"
    
    # Create the new path
    if test_type == "unit":
        new_path = os.path.join(new_root, "unit", category, os.path.basename(file_path))
    else:
        new_path = os.path.join(new_root, "integration", os.path.basename(file_path))
    
    return new_path, test_type, category


def copy_test_file(src, dst, dry_run=True):
    """Copy a test file to its new location."""
    # Skip if source and destination are the same
    if os.path.abspath(src) == os.path.abspath(dst):
        print(f"Skipping {src} (already in correct location)")
        return
    
    if dry_run:
        print(f"Would copy {src} to {dst}")
        return
    
    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    # Copy the file
    shutil.copy2(src, dst)
    print(f"Copied {src} to {dst}")


def migrate_tests(old_root, new_root, dry_run=True):
    """Migrate tests from old structure to new structure."""
    # Find all test files in the old structure
    test_files = find_test_files(old_root)
    
    # Determine new locations for each test file
    migrations = []
    for file_path in test_files:
        new_path, test_type, category = determine_new_location(file_path, old_root, new_root)
        migrations.append((file_path, new_path, test_type, category))
    
    # Print migration plan
    print("\nMigration Plan:")
    print("===============")
    
    for old_path, new_path, test_type, category in migrations:
        rel_old = os.path.relpath(old_path, os.path.dirname(old_root))
        rel_new = os.path.relpath(new_path, new_root)
        print(f"{rel_old} -> {rel_new} ({test_type}/{category})")
    
    # Execute migrations if not a dry run
    if not dry_run:
        print("\nExecuting migrations...")
        for old_path, new_path, _, _ in migrations:
            copy_test_file(old_path, new_path, dry_run=False)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate tests to the new structure")
    parser.add_argument("--old", default=None, help="Old test directory path")
    parser.add_argument("--new", default="MacAgent/tests", help="New test directory path")
    parser.add_argument("--execute", action="store_true", help="Execute migrations (default is dry run)")
    args = parser.parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if args.old is None:
        # Look for both potential old test directories
        old_paths = [
            os.path.join(project_root, "tests"),
            os.path.join(project_root, "MacAgent", "tests")
        ]
        old_paths = [p for p in old_paths if os.path.exists(p) and p != script_dir]
    else:
        old_paths = [os.path.abspath(args.old)]
    
    new_root = os.path.abspath(args.new)
    
    # Migrate each old directory
    for old_root in old_paths:
        print(f"\nMigrating from {old_root} to {new_root}")
        migrate_tests(old_root, new_root, dry_run=not args.execute)


if __name__ == "__main__":
    main() 
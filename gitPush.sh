#!/bin/bash

# Define variables
REPO_DIR="/path/to/your/repo"  # Change this to your repo path
BRANCH="main"  # Change this if your branch is different
# COMMIT_MSG="Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
COMMIT_MSG=${1:-"Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"}

# Navigate to the repo
# cd "$REPO_DIR" || { echo "Repo not found!"; exit 1; }

# Add all changes
git add .

# Commit changes with timestamp
git commit -m "$COMMIT_MSG"

# Push to the remote repository
git push origin "$BRANCH"

# Print success message
echo "Changes pushed to $BRANCH successfully!"

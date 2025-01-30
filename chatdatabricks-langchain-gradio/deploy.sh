#!/usr/bin/env bash

# Set the app name (character limit 30)
APP_NAME="chatdatabricks-langchain"

# Get the current user
CURRENT_USER=$(databricks auth describe | grep "User:" | cut -d' ' -f2)

# Set the workspace path
WORKSPACE_PATH="/Workspace/Users/$CURRENT_USER/databricks-apps-scratchpad/$APP_NAME"

# Sync the current directory to the workspace while respecting the .gitignore file
databricks sync --full . "$WORKSPACE_PATH"
echo "Files synced to workspace path: $WORKSPACE_PATH"

# Deploy the app - assuming the app name exists in the workspace
databricks apps deploy $APP_NAME --source-code-path "$WORKSPACE_PATH"

# Check the status of the deployment
databricks apps get $APP_NAME
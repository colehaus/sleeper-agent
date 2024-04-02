#!/usr/bin/env bash

# Define the remote and branch
REMOTE="origin"
BRANCH="main"

# Function to check for updates and reset
update_repository() {
  # Fetch changes from the remote without altering the working directory
  git fetch --quiet "$REMOTE" "$BRANCH"

  # Check if the local branch is behind the remote branch
  LOCAL_HEAD=$(git rev-parse HEAD)
  REMOTE_HEAD=$(git rev-parse "$REMOTE/$BRANCH")

  if [ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]; then
    echo "Updating and resetting to the latest commit from $REMOTE/$BRANCH..."
    # Reset the current branch to the remote branch --hard
    git reset --hard "$REMOTE/$BRANCH"
  else
    echo "Already up to date with $REMOTE/$BRANCH."
  fi
}

while true; do
  update_repository
  sleep 2
done

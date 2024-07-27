#!/bin/bash

# Show the git log to find the commit hash
echo "Current git log:"
git log --oneline -n 10

# Ask for the commit hash to remove
read -p "Enter the hash of the commit you want to remove: " commit_to_remove

# Get the current branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Create a new branch
new_branch="${current_branch}_clean"
git checkout -b $new_branch

# Get the hash of the previous commit
previous_commit=$(git rev-parse $commit_to_remove^)

# Reset to the commit before the one we want to remove
git reset --hard $previous_commit

# Get all commit hashes after the one we removed
commits_to_cherry_pick=$(git rev-list $commit_to_remove...$current_branch | tac)

# Cherry-pick each commit
for commit in $commits_to_cherry_pick
do
    git cherry-pick $commit
done

echo "The commit has been removed, and all subsequent changes have been re-applied."
echo "You are now on branch $new_branch"

# Offer to force push
read -p "Do you want to force push this new branch? (y/n) " push_confirm
if [ "$push_confirm" = "y" ]; then
    git push origin $new_branch --force
    echo "New branch force pushed."
    echo "You may want to update the original branch to point to this new one:"
    echo "git branch -f $current_branch $new_branch"
    echo "git push origin $current_branch --force"
else
    echo "Changes not pushed. You can push manually when ready."
fi

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

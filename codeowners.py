import gitlab
import os
from uuid import uuid4

# Configuration
GITLAB_URL = "https://gitlab.com"  # Replace with your GitLab instance URL
ACCESS_TOKEN = "your-personal-access-token"  # Replace with your GitLab PAT
GROUP_ID = "your-group-id"  # Replace with your group ID (numeric or path)
FEATURE_BRANCH = "add-codeowners"
TARGET_BRANCH = "master"

# CODEOWNERS file content
CODEOWNERS_CONTENT = """# CODEOWNERS file for all projects under the group
# Defines ownership for specific files and directories
# Documentation: https://docs.gitlab.com/ee/user/project/code_owners.html

# General project files (e.g., README, CI configuration)
* @my-group/general-approvers

# Documentation files require 2 approvals from the docs team
[Documentation][2]
*.md @my-group/docs-team
docs/ @my-group/docs-team

# Frontend code owned by the frontend team
[Frontend]
/app/frontend/ @my-group/frontend-team
**.js @my-group/frontend-team
**.ts @my-group/frontend-team

# Backend code owned by the backend team
[Backend]
/app/backend/ @my-group/backend-team
**.py @my-group/backend-team
**.rb @my-group/backend-team

# Configuration files require 3 approvals from config team
[Configuration][3]
/config/ @my-group/config-team
**.yml @my-group/config-team
**.yaml @my-group/config-team

# Optional section for less critical files (no approval required)
^[Miscellaneous]
/scripts/ @my-group/dev-team
"""

def main():
    # Initialize GitLab client
    gl = gitlab.Gitlab(GITLAB_URL, private_token=ACCESS_TOKEN)

    try:
        # Get group and its projects
        group = gl.groups.get(GROUP_ID)
        projects = group.projects.list(all=True, include_subgroups=True)

        print(f"Found {len(projects)} projects in group {group.name}")

        for project in projects:
            project_id = project.id
            project = gl.projects.get(project_id)
            print(f"\nProcessing project: {project.name}")

            try:
                # Check if CODEOWNERS file already exists
                try:
                    project.files.get(file_path="CODEOWNERS", ref=TARGET_BRANCH)
                    print(f"CODEOWNERS already exists in {project.name}. Skipping.")
                    continue
                except gitlab.exceptions.GitlabGetError:
                    pass  # File doesn't exist, proceed

                # Create feature branch
                try:
                    project.branches.create({"branch": FEATURE_BRANCH, "ref": TARGET_BRANCH})
                    print(f"Created branch {FEATURE_BRANCH} in {project.name}")
                except gitlab.exceptions.GitlabCreateError as e:
                    if "Branch already exists" in str(e):
                        print(f"Branch {FEATURE_BRANCH} already exists in {project.name}")
                    else:
                        raise e

                # Create or update CODEOWNERS file
                commit_data = {
                    "branch": FEATURE_BRANCH,
                    "commit_message": "Add CODEOWNERS file for code review assignments",
                    "actions": [
                        {
                            "action": "create",
                            "file_path": "CODEOWNERS",
                            "content": CODEOWNERS_CONTENT
                        }
                    ]
                }
                project.commits.create(commit_data)
                print(f"Committed CODEOWNERS file to {FEATURE_BRANCH} in {project.name}")

                # Create merge request
                mr_data = {
                    "source_branch": FEATURE_BRANCH,
                    "target_branch": TARGET_BRANCH,
                    "title": "Add CODEOWNERS file",
                    "description": "Adds CODEOWNERS file to define code review responsibilities."
                }
                mr = project.mergerequests.create(mr_data)
                print(f"Created MR {mr.iid} in {project.name}")

                # Configure protected branch
                try:
                    # Check if master is already protected
                    protected_branches = project.protectedbranches.list()
                    master_protected = any(pb.name == TARGET_BRANCH for pb in protected_branches)

                    if not master_protected:
                        project.protectedbranches.create({
                            "name": TARGET_BRANCH,
                            "merge_access_level": gitlab.const.MAINTAINER_ACCESS,
                            "push_access_level": gitlab.const.NO_ACCESS,
                            "code_owner_approval_required": True
                        })
                        print(f"Protected {TARGET_BRANCH} branch in {project.name} with code owner approval")
                    else:
                        # Update existing protected branch to enable code owner approval
                        for pb in protected_branches:
                            if pb.name == TARGET_BRANCH:
                                project.protectedbranches.update(TARGET_BRANCH, {
                                    "code_owner_approval_required": True
                                })
                                print(f"Updated {TARGET_BRANCH} branch protection in {project.name}")
                                break

                except gitlab.exceptions.GitlabHttpError as e:
                    print(f"Failed to configure protected branch in {project.name}: {e}")

            except Exception as e:
                print(f"Error processing project {project.name}: {e}")

    except Exception as e:
        print(f"Error accessing group or projects: {e}")

if __name__ == "__main__":
    main()

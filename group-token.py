#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
import datetime
import subprocess
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import gitlab
import gitlab.exceptions
from contextlib import contextmanager

# Configuration class for settings
@dataclass
class Config:
    access_token: str
    api_url: str
    log_file: Path = Path("./extender.log")
    temp_file: Path = Path("/tmp/gitlab_token_commands.rb")
    expiry_days: int = 30
    batch_size: int = 100
    max_workers: int = 4
    target_groups: tuple = ("snapshot", "release")

# Setup structured logging
def setup_logging(log_file: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create file handler with structured format
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        json=True,
        fmt='%(asctime)s - %(levelname)s - %(message)s - %(context)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@contextmanager
def temp_file_manager(file_path: Path):
    """Context manager for temporary file handling"""
    try:
        yield file_path
    finally:
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as e:
                logging.error(f"Failed to remove temporary file {file_path}: {str(e)}", 
                            extra={"context": "cleanup"})

async def get_gitlab_client(config: Config) -> Optional[gitlab.Gitlab]:
    """Initialize GitLab client with error handling"""
    try:
        client = gitlab.Gitlab(
            config.api_url,
            private_token=config.access_token,
            user_agent='gitlab-access-token-extender/0.2',
            retry_transient_errors=True
        )
        await asyncio.to_thread(client.auth)
        return client
    except gitlab.exceptions.GitlabAuthenticationError as e:
        logging.error(f"Authentication failed: {str(e)}", extra={"context": "gitlab_auth"})
        return None
    except gitlab.exceptions.GitlabError as e:
        logging.error(f"Failed to initialize GitLab client: {str(e)}", extra={"context": "gitlab_init"})
        return None

def execute_rails_commands(commands: List[str], config: Config) -> bool:
    """Execute GitLab Rails commands in batches"""
    if not commands:
        logging.info("No tokens need to be extended.", extra={"context": "execution", "command_count": 0})
        return True

    logging.info(f"Executing {len(commands)} token extension commands", 
                extra={"context": "execution", "command_count": len(commands)})

    try:
        with temp_file_manager(config.temp_file) as temp_file:
            # Write commands in batches
            for i in range(0, len(commands), config.batch_size):
                batch = commands[i:i + config.batch_size]
                with temp_file.open('a') as f:
                    f.write('\n'.join(batch) + '\n')
                
                subprocess.run(
                    ['gitlab-rails', 'runner', str(temp_file)],
                    check=True,
                    capture_output=True,
                    text=True
                )
        logging.info("Successfully extended token expiration dates", 
                    extra={"context": "execution_success"})
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute rails commands: {e.stderr}", 
                     extra={"context": "execution_error", "error": str(e)})
        return False
    except OSError as e:
        logging.error(f"File operation error: {str(e)}", 
                     extra={"context": "file_error", "error": str(e)})
        return False

async def process_tokens(client: gitlab.Gitlab, check_expiry: bool, config: Config) -> List[str]:
    """Process all types of tokens (personal, group, project)"""
    commands = []
    expiry_date = (datetime.datetime.now() + datetime.timedelta(days=config.expiry_days)).strftime('%Y-%m-%d')

    async def process_single_token(token, token_type: str) -> Optional[str]:
        if token.revoked or not token.active:
            return None
            
        try:
            if token_type == "personal":
                user = await asyncio.to_thread(client.users.get, token.user_id)
                if user.state != "active":
                    return None

            expires_soon = token.expires_at and token.expires_at <= expiry_date
            if check_expiry and not expires_soon:
                return None

            if not check_expiry or expires_soon:
                return f"PersonalAccessToken.where(id: {token.id}).update_all(expires_at: 1.year.from_now)"
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"Error processing {token_type} token {token.id}: {str(e)}", 
                         extra={"context": f"{token_type}_token", "token_id": token.id})
            return None

    async def process_token_list(tokens, token_type: str) -> List[str]:
        tasks = [process_single_token(token, token_type) for token in tokens]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [cmd for cmd in results if cmd is not None]

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Personal tokens
        personal_tokens = await asyncio.to_thread(client.personal_access_tokens.list, all=True, iterator=True)
        personal_commands = await process_token_list(personal_tokens, "personal")

        # Group tokens
        group_commands = []
        groups = await asyncio.to_thread(client.groups.list, all=True, iterator=True)
        for group in groups:
            if group.name.lower() not in config.target_groups:
                continue
            try:
                tokens = await asyncio.to_thread(group.access_tokens.list, all=True, iterator=True)
                group_commands.extend(await process_token_list(tokens, "group"))
            except gitlab.exceptions.GitlabError as e:
                logging.error(f"Error accessing tokens for group {group.name}: {str(e)}", 
                            extra={"context": "group_tokens", "group_id": group.id})

        # Project tokens from specific groups
        project_commands = []
        for group in groups:
            if group.name.lower() not in config.target_groups:
                continue
            try:
                projects = await asyncio.to_thread(group.projects.list, all=True, iterator=True)
                for project in projects:
                    try:
                        tokens = await asyncio.to_thread(client.projects.get(project.id).access_tokens.list, 
                                                      all=True, iterator=True)
                        project_commands.extend(await process_token_list(tokens, "project"))
                    except gitlab.exceptions.GitlabError as e:
                        logging.error(f"Error accessing tokens for project {project.name}: {str(e)}", 
                                    extra={"context": "project_tokens", "project_id": project.id})
            except gitlab.exceptions.GitlabError as e:
                logging.error(f"Error accessing projects for group {group.name}: {str(e)}", 
                            extra={"context": "group_projects", "group_id": group.id})

    commands.extend(personal_commands + group_commands + project_commands)
    return commands

async def main():
    """Main function to coordinate token extension process"""
    # Load configuration
    load_dotenv()
    config = Config(
        access_token=os.getenv("GITLAB_ACCESS_TOKEN", ""),
        api_url=os.getenv("GITLAB_API_URL", "")
    )

    # Validate configuration
    if not config.access_token:
        logging.error("GITLAB_ACCESS_TOKEN environment variable is not set", 
                     extra={"context": "config_validation"})
        sys.exit(1)
    if not config.api_url:
        logging.error("GITLAB_API_URL environment variable is not set", 
                     extra={"context": "config_validation"})
        sys.exit(1)

    # Setup logging
    setup_logging(config.log_file)

    # Check command line arguments
    check_expiry = len(sys.argv) > 1 and sys.argv[1] == "--check-expiry"

    # Initialize GitLab client
    client = await get_gitlab_client(config)
    if not client:
        sys.exit(1)

    # Process tokens and execute commands
    try:
        commands = await process_tokens(client, check_expiry, config)
        success = execute_rails_commands(commands, config)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", 
                     extra={"context": "main_execution", "error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
import datetime
import subprocess
import json
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import gitlab
import gitlab.exceptions
from contextlib import contextmanager
from time import perf_counter
from functools import wraps

# Configuration class
@dataclass
class Config:
    access_token: str
    api_url: str
    log_file: Path = Path("./extender.log")
    temp_file: Path = Path("/tmp/gitlab_token_commands.rb")
    expiry_days: int = 30
    batch_size: int = 100
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay: float = 1.0

# Structured logging setup
def setup_logging(log_file: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": record.created,
                "level": record.levelname,
                "message": record.getMessage(),
                "context": getattr(record, "context", {}),
                "filename": record.filename,
                "lineno": record.lineno
            }
            return json.dumps(log_data)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

# Timing decorator for performance metrics
def timeit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = await func(*args, **kwargs)
        duration = perf_counter() - start_time
        logging.info(f"Function {func.__name__} took {duration:.2f} seconds",
                    extra={"context": {"function": func.__name__, "duration": duration}})
        return result
    return wrapper

@contextmanager
def temp_file_manager(file_path: Path):
    """Context manager for temporary file handling"""
    try:
        yield file_path
    finally:
        if file_path.exists():
            try:
                file_path.unlink()
                logging.info("Cleaned up temporary file",
                           extra={"context": {"file_path": str(file_path)}})
            except OSError as e:
                logging.error(f"Failed to remove temporary file: {str(e)}",
                            extra={"context": {"file_path": str(file_path), "error": str(e)}})

async def get_gitlab_client(config: Config) -> Optional[gitlab.Gitlab]:
    """Initialize GitLab client with retry logic"""
    for attempt in range(config.retry_attempts):
        try:
            client = gitlab.Gitlab(
                config.api_url,
                private_token=config.access_token,
                user_agent='gitlab-access-token-extender/0.2',
                retry_transient_errors=True
            )
            await asyncio.to_thread(client.auth)
            logging.info("Successfully authenticated with GitLab",
                        extra={"context": {"api_url": config.api_url}})
            return client
        except gitlab.exceptions.GitlabAuthenticationError as e:
            logging.error(f"Authentication attempt {attempt + 1} failed: {str(e)}",
                         extra={"context": {"attempt": attempt + 1, "error": str(e)}})
            if attempt < config.retry_attempts - 1:
                await asyncio.sleep(config.retry_delay)
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"GitLab client initialization failed: {str(e)}",
                         extra={"context": {"attempt": attempt + 1, "error": str(e)}})
            return None
    logging.error("All authentication attempts failed",
                extra={"context": {"attempts": config.retry_attempts}})
    return None

def execute_rails_commands(commands: List[str], config: Config) -> bool:
    """Execute GitLab Rails commands in batches"""
    if not commands:
        logging.info("No tokens need to be extended",
                    extra={"context": {"command_count": 0}})
        return True

    logging.info(f"Executing {len(commands)} token extension commands",
                extra={"context": {"command_count": len(commands)}})

    try:
        with temp_file_manager(config.temp_file) as temp_file:
            for i in range(0, len(commands), config.batch_size):
                batch = commands[i:i + config.batch_size]
                with temp_file.open('a') as f:
                    f.write('\n'.join(batch) + '\n')
                
                start_time = perf_counter()
                result = subprocess.run(
                    ['gitlab-rails', 'runner', str(temp_file)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                duration = perf_counter() - start_time
                logging.info(f"Executed batch of {len(batch)} commands",
                           extra={"context": {
                               "batch_size": len(batch),
                               "duration": duration,
                               "return_code": result.returncode
                           }})
        logging.info("Successfully extended token expiration dates",
                    extra={"context": {"total_commands": len(commands)}})
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute rails commands: {e.stderr}",
                     extra={"context": {"error": str(e), "stderr": e.stderr}})
        return False
    except OSError as e:
        logging.error(f"File operation error: {str(e)}",
                     extra={"context": {"error": str(e)}})
        return False

@timeit
async def process_tokens(client: gitlab.Gitlab, check_expiry: bool, config: Config) -> List[str]:
    """Process all types of tokens with parallel execution"""
    commands = []
    expiry_date = (datetime.datetime.now() + datetime.timedelta(days=config.expiry_days)).strftime('%Y-%m-%d')
    processed_counts: Dict[str, int] = {"personal": 0, "group": 0, "project": 0}

    async def process_single_token(token, token_type: str) -> Optional[str]:
        try:
            if token.revoked or not token.active:
                logging.debug(f"Skipping inactive/revoked {token_type} token",
                            extra={"context": {"token_id": token.id, "token_type": token_type}})
                return None

            if token_type == "personal":
                user = await asyncio.to_thread(client.users.get, token.user_id)
                if user.state != "active":
                    logging.debug(f"Skipping token for inactive user",
                                extra={"context": {"token_id": token.id, "user_id": token.user_id}})
                    return None

            expires_soon = token.expires_at and token.expires_at <= expiry_date
            if check_expiry and not expires_soon:
                return None

            if not check_expiry or expires_soon:
                processed_counts[token_type] += 1
                logging.debug(f"Extending {token_type} token",
                            extra={"context": {
                                "token_id": token.id,
                                "expires_at": token.expires_at,
                                "new_expiry": "1.year.from_now"
                            }})
                return f"PersonalAccessToken.where(id: {token.id}).update_all(expires_at: 1.year.from_now)"
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"Error processing {token_type} token: {str(e)}",
                         extra={"context": {"token_id": token.id, "token_type": token_type, "error": str(e)}})
            return None

    async def process_token_list(tokens, token_type: str) -> List[str]:
        tasks = [process_single_token(token, token_type) for token in tokens]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [cmd for cmd in results if isinstance(cmd, str)]

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Personal tokens
        start_time = perf_counter()
        personal_tokens = await asyncio.to_thread(client.personal_access_tokens.list, all=True, iterator=True)
        personal_commands = await process_token_list(personal_tokens, "personal")
        logging.info(f"Processed personal tokens",
                    extra={"context": {
                        "token_count": processed_counts["personal"],
                        "duration": perf_counter() - start_time
                    }})

        # Group tokens
        group_commands = []
        start_time = perf_counter()
        groups = await asyncio.to_thread(client.groups.list, all=True, iterator=True)
        for group in groups:
            try:
                tokens = await asyncio.to_thread(group.access_tokens.list, all=True, iterator=True)
                group_commands.extend(await process_token_list(tokens, "group"))
            except gitlab.exceptions.GitlabError as e:
                logging.error(f"Error accessing tokens for group: {str(e)}",
                            extra={"context": {"group_id": group.id, "group_name": group.name, "error": str(e)}})

        logging.info(f"Processed group tokens",
                    extra={"context": {
                        "token_count": processed_counts["group"],
                        "duration": perf_counter() - start_time
                    }})

        # Project tokens
        project_commands = []
        start_time = perf_counter()
        projects = await asyncio.to_thread(client.projects.list, all=True, iterator=True)
        for project in projects:
            try:
                tokens = await asyncio.to_thread(project.access_tokens.list, all=True, iterator=True)
                project_commands.extend(await process_token_list(tokens, "project"))
            except gitlab.exceptions.GitlabError as e:
                logging.error(f"Error accessing tokens for project: {str(e)}",
                            extra={"context": {"project_id": project.id, "project_name": project.name, "error": str(e)}})

        logging.info(f"Processed project tokens",
                    extra={"context": {
                        "token_count": processed_counts["project"],
                        "duration": perf_counter() - start_time
                    }})

    commands.extend(personal_commands + group_commands + project_commands)
    logging.info("Completed token processing",
                extra={"context": {
                    "total_tokens": sum(processed_counts.values()),
                    "breakdown": processed_counts
                }})
    return commands

async def main():
    """Main function to coordinate token extension process"""
    start_time = perf_counter()
    
    # Load configuration
    load_dotenv()
    config = Config(
        access_token=os.getenv("GITLAB_ACCESS_TOKEN", ""),
        api_url=os.getenv("GITLAB_API_URL", "")
    )

    # Validate configuration
    if not config.access_token:
        logging.error("GITLAB_ACCESS_TOKEN environment variable is not set",
                     extra={"context": {"validation": "access_token"}})
        sys.exit(1)
    if not config.api_url:
        logging.error("GITLAB_API_URL environment variable is not set",
                     extra={"context": {"validation": "api_url"}})
        sys.exit(1)

    # Setup logging
    setup_logging(config.log_file)
    logging.info("Starting token extension process",
                extra={"context": {"start_time": datetime.datetime.now().isoformat()}})

    # Check command line arguments
    check_expiry = len(sys.argv) > 1 and sys.argv[1] == "--check-expiry"
    logging.info(f"Running with check_expiry: {check_expiry}",
                extra={"context": {"check_expiry": check_expiry}})

    # Initialize GitLab client
    client = await get_gitlab_client(config)
    if not client:
        logging.error("Failed to initialize GitLab client",
                     extra={"context": {"initialization": "failed"}})
        sys.exit(1)

    # Process tokens and execute commands
    try:
        commands = await process_tokens(client, check_expiry, config)
        success = execute_rails_commands(commands, config)
        logging.info("Token extension process completed",
                    extra={"context": {
                        "success": success,
                        "total_duration": perf_counter() - start_time
                    }})
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Unexpected error in main process: {str(e)}",
                     extra={"context": {"error": str(e)}})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

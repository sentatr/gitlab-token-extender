#!/usr/bin/env python3
"""
GitLab Access Token Extender
Manages and extends GitLab access tokens with comprehensive logging and error handling.
"""
import os
import sys
import uuid
import logging
import json
import subprocess
import datetime
from typing import List, Optional
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
from functools import wraps
from dotenv import load_dotenv
import gitlab
import gitlab.exceptions
from dataclasses import dataclass

# Configuration class
@dataclass
class Config:
    """Configuration settings for GitLab token extender"""
    access_token: str
    api_url: str
    log_file: Path = Path("./extender.log")
    temp_file: Path = Path("/tmp/gitlab_token_commands.rb")
    expiry_days: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.access_token:
            raise ValueError("GITLAB_ACCESS_TOKEN is required")
        if not self.api_url:
            raise ValueError("GITLAB_API_URL is required")
        if not self.log_file.parent.exists():
            raise ValueError(f"Log file directory {self.log_file.parent} does not exist")

# Structured JSON logging setup
def setup_logging(log_file: Path) -> None:
    """Configure structured JSON logging with correlation ID"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "correlation_id": getattr(record, "correlation_id", str(uuid.uuid4())),
                "context": getattr(record, "context", {}),
                "filename": record.filename,
                "lineno": record.lineno
            }
            return json.dumps(log_data, ensure_ascii=False)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Timing decorator
def timeit(func):
    """Decorator to measure and log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        correlation_id = str(uuid.uuid4())
        logging.info(f"Starting {func.__name__}",
                     extra={"correlation_id": correlation_id, "context": {"function": func.__name__}})
        try:
            result = func(*args, **kwargs)
            duration = perf_counter() - start_time
            logging.info(f"Completed {func.__name__}",
                         extra={"correlation_id": correlation_id,
                               "context": {"function": func.__name__, "duration": duration}})
            return result
        except Exception as e:
            duration = perf_counter() - start_time
            logging.error(f"Error in {func.__name__}: {str(e)}",
                         extra={"correlation_id": correlation_id,
                               "context": {"function": func.__name__, "duration": duration, "error": str(e)}})
            raise
    return wrapper

@contextmanager
def temp_file_manager(file_path: Path, correlation_id: str):
    """Context manager for temporary file handling with logging"""
    logging.debug("Creating temporary file",
                 extra={"correlation_id": correlation_id, "context": {"file_path": str(file_path)}})
    try:
        yield file_path
    finally:
        if file_path.exists():
            try:
                file_path.unlink()
                logging.info("Cleaned up temporary file",
                            extra={"correlation_id": correlation_id,
                                  "context": {"file_path": str(file_path)}})
            except OSError as e:
                logging.error(f"Failed to remove temporary file: {str(e)}",
                             extra={"correlation_id": correlation_id,
                                   "context": {"file_path": str(file_path), "error": str(e)}})

def get_gitlab_client(config: Config, correlation_id: str) -> Optional[gitlab.Gitlab]:
    """Initialize GitLab client with retry logic"""
    for attempt in range(config.retry_attempts):
        try:
            client = gitlab.Gitlab(
                config.api_url,
                private_token=config.access_token,
                user_agent='gitlab-access-token-extender/0.1',
                retry_transient_errors=True
            )
            client.auth()
            logging.info("Successfully authenticated with GitLab",
                        extra={"correlation_id": correlation_id,
                              "context": {"api_url": config.api_url, "attempt": attempt + 1}})
            return client
        except gitlab.exceptions.GitlabAuthenticationError as e:
            logging.error(f"Authentication attempt {attempt + 1} failed: {str(e)}",
                         extra={"correlation_id": correlation_id,
                               "context": {"attempt": attempt + 1, "error": str(e)}})
            if attempt < config.retry_attempts - 1:
                import time
                time.sleep(config.retry_delay)
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"GitLab client initialization failed: {str(e)}",
                         extra={"correlation_id": correlation_id,
                               "context": {"attempt": attempt + 1, "error": str(e)}})
            return None
    logging.error("All authentication attempts failed",
                 extra={"correlation_id": correlation_id,
                       "context": {"attempts": config.retry_attempts}})
    return None

@timeit
def execute_rails_commands(commands: List[str], config: Config, correlation_id: str) -> bool:
    """Execute GitLab Rails commands"""
    if not commands:
        logging.info("No tokens need to be extended",
                    extra={"correlation_id": correlation_id,
                          "context": {"command_count": 0}})
        return True

    logging.info(f"Executing {len(commands)} token extension commands",
                extra={"correlation_id": correlation_id,
                      "context": {"command_count": len(commands)}})

    try:
        with temp_file_manager(config.temp_file, correlation_id) as temp_file:
            with temp_file.open('w', encoding='utf-8') as f:
                f.write('\n'.join(commands) + '\n')
            
            start_time = perf_counter()
            result = subprocess.run(
                ['gitlab-rails', 'runner', str(temp_file)],
                check=True,
                capture_output=True,
                text=True
            )
            duration = perf_counter() - start_time
            logging.info("Successfully extended token expiration dates",
                        extra={"correlation_id": correlation_id,
                              "context": {
                                  "command_count": len(commands),
                                  "duration": duration,
                                  "return_code": result.returncode,
                                  "stdout": result.stdout[:1000]  # Limit log size
                              }})
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute rails commands: {e.stderr}",
                     extra={"correlation_id": correlation_id,
                           "context": {"error": str(e), "stderr": e.stderr[:1000]}})
        return False
    except OSError as e:
        logging.error(f"File operation error: {str(e)}",
                     extra={"correlation_id": correlation_id,
                           "context": {"error": str(e)}})
        return False

@timeit
def process_tokens(client: gitlab.Gitlab, check_expiry: bool, config: Config, correlation_id: str) -> List[str]:
    """Process all types of tokens and generate extension commands"""
    commands = []
    expiry_date = (datetime.datetime.now() + datetime.timedelta(days=config.expiry_days)).strftime('%Y-%m-%d')
    processed_counts = {"personal": 0, "group": 0, "project": 0}

    def process_single_token(token, token_type: str) -> Optional[str]:
        token_correlation_id = str(uuid.uuid4())
        try:
            if token.revoked or not token.active:
                logging.debug(f"Skipping inactive/revoked {token_type} token",
                            extra={"correlation_id": token_correlation_id,
                                  "context": {"token_id": token.id, "token_type": token_type}})
                return None

            if token_type == "personal":
                user = client.users.get(token.user_id)
                if user.state != "active":
                    logging.debug(f"Skipping token for inactive user",
                                 extra={"correlation_id": token_correlation_id,
                                       "context": {"token_id": token.id, "user_id": token.user_id}})
                    return None

            expires_soon = token.expires_at and token.expires_at <= expiry_date
            if check_expiry and not expires_soon:
                return None

            processed_counts[token_type] += 1
            logging.debug(f"Extending {token_type} token",
                         extra={"correlation_id": token_correlation_id,
                               "context": {
                                   "token_id": token.id,
                                   "expires_at": token.expires_at,
                                   "new_expiry": "1.year.from_now"
                               }})
            return f"PersonalAccessToken.where(id: {token.id}).update_all(expires_at: 1.year.from_now)"
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"Error processing {token_type} token: {str(e)}",
                         extra={"correlation_id": token_correlation_id,
                               "context": {"token_id": token.id, "token_type": token_type, "error": str(e)}})
            return None

    # Process personal tokens
    start_time = perf_counter()
    personal_tokens = client.personal_access_tokens.list(all=True, iterator=True)
    commands.extend([cmd for cmd in map(lambda t: process_single_token(t, "personal"), personal_tokens) if cmd])
    logging.info(f"Processed personal tokens",
                extra={"correlation_id": correlation_id,
                      "context": {"token_count": processed_counts["personal"],
                                 "duration": perf_counter() - start_time}})

    # Process group tokens
    start_time = perf_counter()
    for group in client.groups.list(all=True, iterator=True):
        try:
            tokens = group.access_tokens.list(all=True, iterator=True)
            commands.extend([cmd for cmd in map(lambda t: process_single_token(t, "group"), tokens) if cmd])
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"Error accessing tokens for group: {str(e)}",
                         extra={"correlation_id": correlation_id,
                               "context": {"group_id": group.id, "group_name": group.name, "error": str(e)}})
    logging.info(f"Processed group tokens",
                extra={"correlation_id": correlation_id,
                      "context": {"token_count": processed_counts["group"],
                                 "duration": perf_counter() - start_time}})

    # Process project tokens
    start_time = perf_counter()
    for project in client.projects.list(all=True, iterator=True):
        try:
            tokens = project.access_tokens.list(all=True, iterator=True)
            commands.extend([cmd for cmd in map(lambda t: process_single_token(t, "project"), tokens) if cmd])
        except gitlab.exceptions.GitlabError as e:
            logging.error(f"Error accessing tokens for project: {str(e)}",
                         extra={"correlation_id": correlation_id,
                               "context": {"project_id": project.id, "project_name": project.name, "error": str(e)}})
    logging.info(f"Processed project tokens",
                extra={"correlation_id": correlation_id,
                      "context": {"token_count": processed_counts["project"],
                                 "duration": perf_counter() - start_time}})

    logging.info("Completed token processing",
                extra={"correlation_id": correlation_id,
                      "context": {"total_tokens": sum(processed_counts.values()),
                                 "breakdown": processed_counts}})
    return commands

def main():
    """Main function to coordinate token extension process"""
    correlation_id = str(uuid.uuid4())
    start_time = perf_counter()

    # Load and validate configuration
    load_dotenv()
    config = Config(
        access_token=os.getenv("GITLAB_ACCESS_TOKEN", ""),
        api_url=os.getenv("GITLAB_API_URL", "")
    )
    
    try:
        config.validate()
    except ValueError as e:
        logging.error(f"Configuration validation failed: {str(e)}",
                     extra={"correlation_id": correlation_id,
                           "context": {"error": str(e)}})
        sys.exit(1)

    # Setup logging
    setup_logging(config.log_file)
    logging.info("Starting token extension process",
                extra={"correlation_id": correlation_id,
                      "context": {"start_time": datetime.datetime.now().isoformat()}})

    # Check command line arguments
    check_expiry = len(sys.argv) > 1 and sys.argv[1] == "--check-expiry"
    logging.info(f"Running with check_expiry: {check_expiry}",
                extra={"correlation_id": correlation_id,
                      "context": {"check_expiry": check_expiry}})

    # Initialize GitLab client
    client = get_gitlab_client(config, correlation_id)
    if not client:
        logging.error("Failed to initialize GitLab client",
                     extra={"correlation_id": correlation_id,
                           "context": {"initialization": "failed"}})
        sys.exit(1)

    # Process tokens and execute commands
    try:
        commands = process_tokens(client, check_expiry, config, correlation_id)
        success = execute_rails_commands(commands, config, correlation_id)
        logging.info("Token extension process completed",
                    extra={"correlation_id": correlation_id,
                          "context": {
                              "success": success,
                              "total_duration": perf_counter() - start_time,
                              "total_tokens": len(commands)
                          }})
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Unexpected error in main process: {str(e)}",
                     extra={"correlation_id": correlation_id,
                           "context": {"error": str(e)}})
        sys.exit(1)

if __name__ == "__main__":
    main()

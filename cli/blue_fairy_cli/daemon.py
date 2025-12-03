"""Supervisor daemon management"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional

from .client import SupervisorClient


class DaemonManager:
    """Manage supervisor daemon lifecycle"""

    def __init__(self, blue_fairy_dir: Path = None):
        if blue_fairy_dir is None:
            blue_fairy_dir = Path.home() / ".blue-fairy"

        self.blue_fairy_dir = blue_fairy_dir
        self.blue_fairy_dir.mkdir(parents=True, exist_ok=True)

        self.pid_file = self.blue_fairy_dir / "supervisor.pid"
        self.log_file = self.blue_fairy_dir / "supervisor.log"
        self.db_file = self.blue_fairy_dir / "state.db"
        self.config_file = self.blue_fairy_dir / "config.yaml"

    def is_running(self) -> bool:
        """Check if supervisor is running"""
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # Process doesn't exist or PID file is invalid
            self.pid_file.unlink(missing_ok=True)
            return False

    def start(self) -> bool:
        """Start supervisor daemon"""
        if self.is_running():
            return True

        # Prepare environment variables
        env = os.environ.copy()
        env["DB_PATH"] = str(self.db_file)
        env["CONFIG_PATH"] = str(self.config_file)
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # Prevent bytecode caching for development

        # Ensure Docker socket matches CLI context (for Docker Desktop)
        if "DOCKER_HOST" not in env:
            docker_desktop_sock = Path.home() / ".docker" / "desktop" / "docker.sock"
            if docker_desktop_sock.exists():
                env["DOCKER_HOST"] = f"unix://{docker_desktop_sock}"

        # Start supervisor process in background
        log_file = open(self.log_file, "w")

        cmd = [
            sys.executable, "-B", "-m", "uvicorn",  # -B flag prevents bytecode generation
            "blue_fairy_supervisor.api:create_app_from_env",
            "--factory",
            "--host", "0.0.0.0",  # Listen on all interfaces so Docker containers can reach supervisor
            "--port", "8765",
        ]

        # Launch subprocess in background
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from parent
        )

        # Write PID file
        self.pid_file.write_text(str(process.pid))

        # Don't wait for process, let it run in background
        # The log file handle will be kept open by the subprocess

        return True

    def stop(self) -> bool:
        """Stop supervisor daemon"""
        if not self.is_running():
            return False

        try:
            pid = int(self.pid_file.read_text())
            os.kill(pid, signal.SIGTERM)

            # Wait for process to exit
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break

            self.pid_file.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def ensure_running(self, timeout: int = 10) -> bool:
        """Ensure supervisor is running, start if needed"""
        if self.is_running():
            # Verify it's actually responsive
            client = SupervisorClient()
            if client.health_check():
                return True

        # Start daemon
        if not self.start():
            return False

        # Wait for health check
        client = SupervisorClient()
        for _ in range(timeout * 2):
            if client.health_check():
                return True
            time.sleep(0.5)

        return False

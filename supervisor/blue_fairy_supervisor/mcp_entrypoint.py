"""Entry point for running Blue Fairy MCP server."""

import asyncio
import logging
from pathlib import Path

from blue_fairy_supervisor.state import StateManager
from blue_fairy_supervisor.matrix_manager import MatrixManager
from blue_fairy_supervisor.docker_mgr import DockerManager
from blue_fairy_supervisor.observability import ObservabilityManager
from blue_fairy_supervisor.mcp_server import BlueFairyMCPServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Initialize supervisor components and run MCP server."""
    # Use default state directory
    state_dir = Path.home() / ".blue-fairy"
    state_dir.mkdir(exist_ok=True)

    db_path = state_dir / "state.db"
    obs_db_path = state_dir / "observability.db"

    # Initialize components
    state_manager = StateManager(str(db_path))
    docker_manager = DockerManager()
    matrix_manager = MatrixManager(state_db_path=db_path)

    # Initialize observability manager (optional, may not exist yet)
    observability_manager = None
    try:
        observability_manager = ObservabilityManager(obs_db_path)
        logger.info("Observability manager initialized")
    except Exception as e:
        logger.warning(f"Observability manager not available: {e}")

    # Create and run MCP server
    mcp_server = BlueFairyMCPServer(
        state_manager,
        matrix_manager,
        docker_manager,
        observability_manager
    )

    logger.info("Starting Blue Fairy MCP server...")
    await mcp_server.run()


if __name__ == "__main__":
    asyncio.run(main())

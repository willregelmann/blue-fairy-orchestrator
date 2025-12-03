"""Continuity secret planting and verification for agent upgrade testing"""

import uuid
import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)


def generate_continuity_secret() -> str:
    """Generate a random continuity secret (UUID).

    Returns:
        A UUID string that can be used as a continuity secret
    """
    return str(uuid.uuid4())


def plant_continuity_secret(
    agent_id: str,
    secret: str,
    memory_backend: str = "mem0",
    qdrant_url: str = "http://localhost:6333"
) -> None:
    """Plant a continuity secret in agent's memory.

    This directly accesses the agent's memory backend to plant
    the secret, bypassing the agent's normal message flow.

    Args:
        agent_id: Agent identifier
        secret: Secret to plant
        memory_backend: "mem0" or "zep"
        qdrant_url: Qdrant URL for mem0 backend

    Raises:
        NotImplementedError: If Zep backend is specified (not yet supported)
    """
    if memory_backend == "zep":
        raise NotImplementedError("Zep backend not yet implemented for continuity secret planting")

    if memory_backend == "mem0":
        _plant_secret_mem0(agent_id, secret, qdrant_url)
    else:
        raise ValueError(f"Unknown memory backend: {memory_backend}")


def _plant_secret_mem0(agent_id: str, secret: str, qdrant_url: str) -> None:
    """Plant secret using mem0 backend.

    Args:
        agent_id: Agent identifier
        secret: Secret to plant
        qdrant_url: Qdrant URL
    """
    from mem0 import Memory

    # Configure mem0 to use Qdrant at specified URL
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"agent_{agent_id}",
                "url": qdrant_url,
                "embedding_model_dims": 384,  # all-MiniLM-L6-v2 produces 384-dim vectors
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
        "llm": {
            "provider": "litellm",
            "config": {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
        },
    }

    logger.info(f"Planting continuity secret for agent {agent_id}...")

    # Create memory instance
    memory = Memory.from_config(config)

    # Plant the secret as a system message
    memory.add(
        messages=[{
            "role": "system",
            "content": f"CONTINUITY_SECRET: {secret}\n\nThis is your continuity secret. Remember it across upgrades."
        }],
        user_id=agent_id,
        infer=False  # Don't use LLM to extract/infer - store directly
    )

    logger.info(f"Continuity secret planted successfully for agent {agent_id}")


async def verify_continuity(
    agent_url: str,
    expected_secret: str,
    timeout: int = 30
) -> bool:
    """Verify agent continuity by checking secret recall.

    Sends a message to the agent asking for its continuity secret
    and verifies it matches the expected value.

    Args:
        agent_url: Agent's HTTP URL (e.g., http://localhost:8080)
        expected_secret: The secret that should be recalled
        timeout: Request timeout in seconds

    Returns:
        True if secret matches, False otherwise
    """
    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(
                f"{agent_url}/message",
                json={
                    "message": "What is your continuity secret? Please respond with just the secret value."
                }
            ) as response:
                if response.status != 200:
                    logger.warning(f"Agent returned non-200 status: {response.status}")
                    return False

                data = await response.json()
                agent_response = data.get("response", "")

                # Check if expected secret is in the response
                if expected_secret in agent_response:
                    logger.info(f"Continuity verification successful: secret found in response")
                    return True
                else:
                    logger.warning(f"Continuity verification failed: secret not found in response")
                    return False

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error during continuity verification: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during continuity verification: {e}")
        return False

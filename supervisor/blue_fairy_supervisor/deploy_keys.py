"""Deploy key management (stub - not yet implemented)"""

from typing import Tuple


def generate_ssh_keypair(agent_id: str) -> Tuple[str, str]:
    """Generate SSH keypair for agent (stub)

    Args:
        agent_id: Agent identifier

    Returns:
        Tuple of (private_key, public_key)
    """
    raise NotImplementedError("Deploy key generation not yet implemented")


class GitHubDeployKeyManager:
    """GitHub deploy key management (stub - not yet implemented)"""

    def __init__(self):
        raise NotImplementedError("GitHub deploy key manager not yet implemented")

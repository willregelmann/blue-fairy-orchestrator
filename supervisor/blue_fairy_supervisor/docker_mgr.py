"""Docker container management"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import docker
from docker.models.containers import Container


def get_docker_client():
    """Get Docker client using the active Docker context.

    This handles Docker Desktop on Linux where the active context
    uses a different socket than the default /var/run/docker.sock.
    """
    try:
        # Get the active Docker context's endpoint
        result = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            endpoint = result.stdout.strip()
            if endpoint and endpoint != "unix:///var/run/docker.sock":
                # Use the custom endpoint
                return docker.DockerClient(base_url=endpoint)
    except Exception:
        pass

    # Fall back to default
    return docker.from_env()


class DockerManager:
    """Manage Docker containers for agents"""

    NETWORK_NAME = "blue-fairy-net"

    def __init__(self, harness_path: Path = None, harness_repo_url: str = None):
        self.client = get_docker_client()

        if harness_path is None:
            # Default to harness/ directory relative to project root
            harness_path = Path(__file__).parent.parent.parent / "harness"

        self.harness_path = harness_path
        self.harness_repo_url = harness_repo_url or "https://github.com/willregelmann/blue-fairy-harness.git"

    def ensure_network(self):
        """Ensure blue-fairy network exists"""
        networks = self.client.networks.list(names=[self.NETWORK_NAME])

        if not networks:
            self.client.networks.create(self.NETWORK_NAME, driver="bridge")

    def ensure_image(self, image: str) -> bool:
        """Ensure image exists locally, pulling from registry if needed.

        Args:
            image: Docker image name with optional tag (e.g., "nginx:latest",
                   "myregistry.io/myimage:v1.2.3")

        Returns:
            True if image is available

        Raises:
            docker.errors.ImageNotFound: If image not found locally or in registry
        """
        import docker.errors

        try:
            self.client.images.get(image)
            return True
        except docker.errors.ImageNotFound:
            # Try pulling from registry
            self.client.images.pull(image)
            return True

    def run_container(
        self,
        image: str,
        name: str,
        memory: Optional[str] = None,
        cpu_quota: Optional[int] = None,
        environment: Optional[dict] = None,
        memory_dir: Optional[Path] = None
    ) -> Container:
        """Run agent container"""
        # Prepare volumes
        volumes = {}

        # Mount memory directory if provided
        if memory_dir:
            volumes[str(memory_dir.absolute())] = {
                'bind': '/data/memory',
                'mode': 'rw'
            }

        # Prepare environment
        env = environment or {}

        # Resource limits
        kwargs = {
            'image': image,
            'name': name,
            'environment': env,
            'network': self.NETWORK_NAME,  # Create container directly on blue-fairy-net
            'ports': {'8080/tcp': None},   # Publish port 8080 to random host port
            'extra_hosts': {'host.docker.internal': 'host-gateway'},  # Allow containers to reach host
            'detach': True,
            'remove': False,
            'labels': {
                'com.docker.compose.project': 'blue-fairy',  # Group in Docker Desktop
            },
        }

        if volumes:
            kwargs['volumes'] = volumes

        if memory:
            kwargs['mem_limit'] = memory

        if cpu_quota:
            kwargs['cpu_quota'] = cpu_quota

        # Create and start container on blue-fairy network
        container = self.client.containers.run(**kwargs)

        return container

    def stop_container(self, container_id: str):
        """Stop container (preserves container for restart)"""
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=10)
        except docker.errors.NotFound:
            pass

    def start_container(self, container_id: str) -> str:
        """Start a stopped container and return its new IP address"""
        try:
            container = self.client.containers.get(container_id)
            container.start()

            # Wait a moment for network to be ready
            import time
            time.sleep(1)

            # Refresh container info and get IP
            container.reload()
            ip_address = container.attrs['NetworkSettings']['Networks']['blue-fairy-net']['IPAddress']

            return ip_address
        except docker.errors.NotFound:
            raise Exception(f"Container {container_id} not found")
        except KeyError:
            raise Exception(f"Container {container_id} not connected to blue-fairy-net")

    def remove_container(self, container_id: str):
        """Remove container (destroys container and its data)"""
        try:
            container = self.client.containers.get(container_id)

            # Stop if running
            if container.status == "running":
                container.stop(timeout=10)

            # Remove container
            container.remove()
        except docker.errors.NotFound:
            pass

    def get_container_status(self, container_id: str) -> Optional[str]:
        """Get container status"""
        try:
            container = self.client.containers.get(container_id)
            return container.status
        except docker.errors.NotFound:
            return None

    def container_is_healthy(self, container_id: str) -> bool:
        """Check if container is running and healthy"""
        status = self.get_container_status(container_id)
        return status == "running"

    def get_container_stats(self, container_id: str) -> dict:
        """Get resource usage stats for a container.

        Args:
            container_id: Docker container ID

        Returns:
            Dict with cpu_percent, memory_usage_mb, memory_limit_mb

        Raises:
            docker.errors.NotFound: Container doesn't exist
        """
        container = self.client.containers.get(container_id)
        stats = container.stats(stream=False)

        # Calculate CPU percentage
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                    stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                       stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0

        # Memory usage
        memory_usage_mb = stats["memory_stats"]["usage"] / (1024 * 1024)
        memory_limit_mb = stats["memory_stats"]["limit"] / (1024 * 1024)

        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_usage_mb": round(memory_usage_mb, 2),
            "memory_limit_mb": round(memory_limit_mb, 2)
        }

    def get_network_status(self) -> dict:
        """Get status of blue-fairy-net network.

        Returns:
            Dict with exists flag and network name
        """
        try:
            network = self.client.networks.get("blue-fairy-net")
            return {
                "exists": True,
                "name": network.name,
                "driver": network.attrs["Driver"]
            }
        except docker.errors.NotFound:
            return {
                "exists": False,
                "name": "blue-fairy-net"
            }

    def get_network_gateway(self) -> str:
        """Get the gateway IP address of the blue-fairy-net network.

        This is the IP address containers can use to reach the host.

        Returns:
            Gateway IP address (e.g., "172.18.0.1")

        Raises:
            RuntimeError: If network doesn't exist or gateway can't be determined
        """
        try:
            network = self.client.networks.get("blue-fairy-net")
            # Get the IPAM config to find the gateway
            ipam_config = network.attrs.get("IPAM", {}).get("Config", [])
            if ipam_config and len(ipam_config) > 0:
                gateway = ipam_config[0].get("Gateway")
                if gateway:
                    return gateway

            # Fallback: try to extract from network inspect
            # Sometimes gateway is in the Containers section or elsewhere
            raise RuntimeError("Could not determine network gateway from IPAM config")

        except docker.errors.NotFound:
            raise RuntimeError("blue-fairy-net network does not exist")

    def build_agent_image(
        self,
        agent_id: str,
        commit_sha: str,
        version: int
    ) -> str:
        """Build a Docker image from an agent's harness branch.

        Args:
            agent_id: Agent identifier
            commit_sha: Git commit SHA to build from
            version: Version number for tagging

        Returns:
            Docker image tag (e.g., "harness:agent-001-v1")

        Raises:
            RuntimeError: If git operations fail
            Exception: If Docker build fails
        """
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone the agent's branch
            branch_name = f"agent/{agent_id}"
            clone_cmd = [
                "git", "clone",
                "--branch", branch_name,
                "--single-branch",
                self.harness_repo_url,
                tmpdir
            ]

            result = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")

            # Checkout specific commit
            checkout_cmd = [
                "git", "-C", tmpdir,
                "checkout", commit_sha
            ]

            result = subprocess.run(
                checkout_cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Git checkout failed: {result.stderr}")

            # Build Docker image
            image_tag = f"harness:{agent_id}-v{version}"
            self.client.images.build(
                path=tmpdir,
                tag=image_tag,
                rm=True
            )

            return image_tag

    def tag_image_status(self, image_tag: str, status: str):
        """Tag an image with its continuity test status.

        Args:
            image_tag: Current image tag (e.g., "harness:agent-001-v1")
            status: Status to append ("passed" or "failed")

        Example:
            tag_image_status("harness:agent-001-v1", "passed")
            # Creates tag: harness:agent-001-v1-passed
        """
        # Get the image
        image = self.client.images.get(image_tag)

        # Parse repo and tag from image_tag
        repo, tag = image_tag.split(":", 1)

        # Create new tag with status appended
        new_tag = f"{tag}-{status}"

        # Tag the image
        image.tag(repo, new_tag)

# supervisor/blue_fairy_supervisor/matrix_manager.py
"""Matrix homeserver management"""

import asyncio
import hmac
import hashlib
import re
import secrets
import string
import docker
import httpx
from pathlib import Path
from typing import List, Optional, Tuple
from docker.models.containers import Container
from nio import AsyncClient, LoginResponse, RoomVisibility, RoomPreset

from .docker_mgr import get_docker_client


class MatrixManager:
    """Manage Synapse homeserver and Matrix operations"""

    SYNAPSE_IMAGE = "matrixdotorg/synapse:latest"
    SYNAPSE_PORT = 8008

    def __init__(self, matrix_dir: Optional[Path] = None, state_db_path: Optional[Path] = None, docker_client=None):
        """Initialize Matrix manager

        Args:
            matrix_dir: Directory for Matrix data (defaults to ~/.blue-fairy/matrix)
            state_db_path: Path to state database (defaults to ~/.blue-fairy/state.db)
            docker_client: Docker client (defaults to docker.from_env())
        """
        if matrix_dir is None:
            matrix_dir = Path.home() / ".blue-fairy" / "matrix"

        if state_db_path is None:
            state_db_path = Path.home() / ".blue-fairy" / "state.db"

        self.matrix_dir = Path(matrix_dir)
        self.matrix_dir.mkdir(parents=True, exist_ok=True)

        self.state_db_path = Path(state_db_path)

        self.config_dir = self.matrix_dir / "config"
        self.data_dir = self.matrix_dir / "data"

        self.config_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Make data directory writable by Synapse container user (UID 991)
        # Using 0o777 to allow the container to write logs, DB, etc.
        self.data_dir.chmod(0o777)

        # Create Synapse data subdirectories with world-writable permissions
        (self.data_dir / "media_store").mkdir(exist_ok=True, mode=0o777)
        (self.data_dir / "uploads").mkdir(exist_ok=True, mode=0o777)

        # Docker client
        self.docker = docker_client or get_docker_client()

        # Supervisor Matrix account
        self.supervisor_user_id: Optional[str] = None
        self.supervisor_access_token: Optional[str] = None
        self.registration_secret: Optional[str] = None
        self.client: Optional[AsyncClient] = None
        self.homeserver_name = "blue-fairy.local"
        self.homeserver_url: Optional[str] = None

    def load_supervisor_credentials(self):
        """Load existing supervisor credentials from database if available"""
        from .state import StateManager

        state = StateManager(self.state_db_path)
        creds = state.get_supervisor_matrix_credentials()
        state.close()

        if creds:
            self.supervisor_user_id = creds['user_id']
            self.supervisor_access_token = creds['access_token']
            self.registration_secret = creds.get('registration_secret')
            return True
        return False

    def get_homeserver_url(self) -> str:
        """Get the homeserver URL, deriving from Docker if not cached.

        Returns:
            Homeserver URL (e.g., http://localhost:41693)

        Raises:
            RuntimeError: If Synapse container not found or not running
        """
        if self.homeserver_url:
            return self.homeserver_url

        # Get from Docker
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            if container.status != "running":
                raise RuntimeError("Synapse container is not running")

            # Get the host port mapping for 8008
            ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            port_bindings = ports.get("8008/tcp", [])
            if not port_bindings:
                raise RuntimeError("Synapse port 8008 not exposed")

            host_port = port_bindings[0]["HostPort"]
            self.homeserver_url = f"http://localhost:{host_port}"
            return self.homeserver_url

        except Exception as e:
            raise RuntimeError(f"Failed to get homeserver URL: {e}")

    def generate_synapse_config(self) -> Tuple[Path, Path, str]:
        """Generate Synapse homeserver configuration

        Returns:
            Tuple of (config_path, log_config_path, registration_secret)
        """
        config_path = self.config_dir / "homeserver.yaml"

        # Check if config already exists and read registration secret from it
        registration_secret = None
        if config_path.exists():
            import yaml
            try:
                with open(config_path) as f:
                    existing_config = yaml.safe_load(f)
                    registration_secret = existing_config.get('registration_shared_secret')
                    if registration_secret:
                        print(f"✓ Using existing registration secret from config")
            except Exception as e:
                print(f"Warning: Could not read existing config: {e}")

        # Generate new registration secret if not found
        if not registration_secret:
            alphabet = string.ascii_letters + string.digits
            registration_secret = ''.join(secrets.choice(alphabet) for _ in range(64))
            print("✓ Generated new registration secret")

        # Store secret for later use
        self.registration_secret = registration_secret

        # Generate signing key (Ed25519)
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        import base64

        signing_key_path = self.data_dir / "blue-fairy.local.signing.key"
        if not signing_key_path.exists():
            private_key = ed25519.Ed25519PrivateKey.generate()
            # Synapse uses its own format for signing keys
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
            # Synapse format: "ed25519 <version> <base64_key>"
            key_b64 = base64.b64encode(private_bytes).decode('ascii')
            signing_key_content = f"ed25519 1 {key_b64}\n"
            signing_key_path.write_text(signing_key_content)
            # Make readable by container user (Synapse runs as UID 991)
            signing_key_path.chmod(0o644)

        # Load templates
        template_dir = Path(__file__).parent / "templates"
        homeserver_template = (template_dir / "homeserver.yaml.template").read_text()
        log_template = (template_dir / "log_config.yaml.template").read_text()

        # Fill in template
        homeserver_config = homeserver_template.format(
            registration_secret=registration_secret
        )

        # Write config files
        config_path = self.config_dir / "homeserver.yaml"
        log_config_path = self.config_dir / "log_config.yaml"

        config_path.write_text(homeserver_config)
        log_config_path.write_text(log_template)

        return config_path, log_config_path, registration_secret

    def spawn_synapse(self) -> str:
        """Spawn Synapse container

        Returns:
            Container ID

        Raises:
            Exception if container already exists
        """
        # Check if container already exists
        try:
            existing = self.docker.containers.get("blue-fairy-synapse")
            if existing.status == "running":
                return existing.id
            # If container exists but not running, remove it
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass

        # Ensure blue-fairy-net network exists
        networks = self.docker.networks.list(names=["blue-fairy-net"])
        if not networks:
            self.docker.networks.create("blue-fairy-net", driver="bridge")

        # Prepare volumes
        volumes = {
            str(self.config_dir.absolute()): {
                'bind': '/config',
                'mode': 'ro'
            },
            str(self.data_dir.absolute()): {
                'bind': '/data',
                'mode': 'rw'
            }
        }

        # Labels for Docker Desktop grouping
        labels = {
            'com.docker.compose.project': 'blue-fairy',
        }

        # Run Synapse container (with retry in case of stale container)
        try:
            container = self.docker.containers.run(
                image=self.SYNAPSE_IMAGE,
                name="blue-fairy-synapse",
                command=["run", "-c", "/config/homeserver.yaml"],
                volumes=volumes,
                network="blue-fairy-net",
                ports={f'{self.SYNAPSE_PORT}/tcp': None},  # Random host port
                detach=True,
                remove=False,
                labels=labels,
            )
        except docker.errors.APIError as e:
            if "Conflict" in str(e):
                # Try to forcefully remove any stale container
                try:
                    stale = self.docker.containers.get("blue-fairy-synapse")
                    stale.remove(force=True)
                except:
                    pass
                # Retry once
                container = self.docker.containers.run(
                    image=self.SYNAPSE_IMAGE,
                    name="blue-fairy-synapse",
                    command=["run", "-c", "/config/homeserver.yaml"],
                    volumes=volumes,
                    network="blue-fairy-net",
                    ports={f'{self.SYNAPSE_PORT}/tcp': None},
                    detach=True,
                    remove=False,
                    labels=labels,
                )
            else:
                raise

        return container.id

    def create_supervisor_account(self) -> Tuple[str, str]:
        """Create supervisor admin account using registration shared secret

        Returns:
            Tuple of (username, password)

        Raises:
            Exception if Synapse is not running or account creation fails
        """
        if not self.registration_secret:
            raise ValueError("Must call generate_synapse_config() first")

        # Generate secure random password
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(48))

        username = "supervisor"

        # Get Synapse URL (via published port)
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            container.reload()  # Refresh container attributes
            port_bindings = container.attrs['NetworkSettings']['Ports'].get(f'{self.SYNAPSE_PORT}/tcp')
            if not port_bindings or not port_bindings[0]:
                raise Exception("Synapse port not published")
            host_port = port_bindings[0]['HostPort']
            homeserver_url = f"http://localhost:{host_port}"
        except docker.errors.NotFound:
            raise Exception("Synapse container not running")

        # Register account using admin registration endpoint
        # This uses the registration shared secret
        register_url = f"{homeserver_url}/_synapse/admin/v1/register"

        # Get nonce
        nonce_response = httpx.get(register_url)
        nonce_response.raise_for_status()
        nonce = nonce_response.json()["nonce"]

        # Compute HMAC for registration
        mac = hmac.new(
            key=self.registration_secret.encode('utf-8'),
            digestmod=hashlib.sha1,
        )

        mac.update(nonce.encode('utf-8'))
        mac.update(b"\x00")
        mac.update(username.encode('utf-8'))
        mac.update(b"\x00")
        mac.update(password.encode('utf-8'))
        mac.update(b"\x00")
        mac.update(b"admin")  # Make this user an admin

        # Register user
        register_data = {
            "nonce": nonce,
            "username": username,
            "password": password,
            "admin": True,
            "mac": mac.hexdigest(),
        }

        register_response = httpx.post(register_url, json=register_data)
        register_response.raise_for_status()

        user_id = register_response.json()["user_id"]

        # Now log in to get access token (run async method synchronously)
        self.homeserver_url = homeserver_url
        asyncio.run(self._login_supervisor(homeserver_url, username, password))

        # Store credentials in database for future use
        from .state import StateManager
        state = StateManager(self.state_db_path)
        state.store_supervisor_matrix_credentials(
            user_id=self.supervisor_user_id,
            access_token=self.supervisor_access_token,
            registration_secret=self.registration_secret
        )
        state.close()

        return username, password

    async def create_supervisor_account_async(self) -> tuple[str, str]:
        """Create supervisor admin account (async version for use in async contexts)

        Returns:
            Tuple of (username, password)

        Raises:
            Exception if Synapse is not running or account creation fails
        """
        if not self.registration_secret:
            raise ValueError("Must call generate_synapse_config() first")

        # Generate secure random password
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(48))

        username = "supervisor"

        # Get Synapse URL (via published port)
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            container.reload()  # Refresh container attributes
            port_bindings = container.attrs['NetworkSettings']['Ports'].get(f'{self.SYNAPSE_PORT}/tcp')
            if not port_bindings or not port_bindings[0]:
                raise Exception("Synapse port not published")
            host_port = port_bindings[0]['HostPort']
            homeserver_url = f"http://localhost:{host_port}"
        except docker.errors.NotFound:
            raise Exception("Synapse container not running")

        # Register account using admin registration endpoint
        register_url = f"{homeserver_url}/_synapse/admin/v1/register"

        # Get nonce
        nonce_response = httpx.get(register_url)
        nonce_response.raise_for_status()
        nonce = nonce_response.json()["nonce"]

        # Compute HMAC for registration
        mac = hmac.new(
            key=self.registration_secret.encode('utf-8'),
            digestmod=hashlib.sha1,
        )

        mac.update(nonce.encode('utf-8'))
        mac.update(b"\x00")
        mac.update(username.encode('utf-8'))
        mac.update(b"\x00")
        mac.update(password.encode('utf-8'))
        mac.update(b"\x00")
        mac.update(b"admin")  # Make this user an admin

        # Register user
        register_data = {
            "nonce": nonce,
            "username": username,
            "password": password,
            "admin": True,
            "mac": mac.hexdigest(),
        }

        register_response = httpx.post(register_url, json=register_data)
        register_response.raise_for_status()

        user_id = register_response.json()["user_id"]

        # Now log in to get access token (await directly since we're in async context)
        self.homeserver_url = homeserver_url
        await self._login_supervisor(homeserver_url, username, password)

        # Store credentials in database for future use
        from .state import StateManager
        state = StateManager(self.state_db_path)
        state.store_supervisor_matrix_credentials(
            user_id=self.supervisor_user_id,
            access_token=self.supervisor_access_token,
            registration_secret=self.registration_secret
        )
        state.close()

        return username, password

    async def _login_supervisor(self, homeserver_url: str, username: str, password: str):
        """Log in supervisor account and store access token"""
        client = AsyncClient(homeserver_url, f"@{username}:{self.homeserver_name}")

        response = await client.login(password)

        if isinstance(response, LoginResponse):
            self.supervisor_user_id = response.user_id
            self.supervisor_access_token = response.access_token
        else:
            raise Exception(f"Login failed: {response}")

        # Close the client since we'll create new ones as needed
        await client.close()

    async def create_agent_account(self, agent_id: str) -> tuple[str, str]:
        """Create a Matrix account for an agent

        Args:
            agent_id: Agent identifier (e.g., "researcher-a4f2")

        Returns:
            Tuple of (matrix_user_id, access_token)
        """
        if not self.registration_secret:
            raise RuntimeError("Registration secret not available. Call generate_synapse_config() first.")

        # Generate username from agent_id (already sanitized)
        username = agent_id

        # Check for username conflicts and append suffix if needed
        from blue_fairy_supervisor.state import StateManager
        with StateManager(self.state_db_path) as state:
            counter = 1
            original_username = username

            while True:
                # Check if this username already exists in database
                cursor = state.conn.cursor()
                cursor.execute(
                    "SELECT agent_id FROM matrix_accounts WHERE matrix_user_id = ?",
                    (f"@{username}:{self.homeserver_name}",)
                )
                if not cursor.fetchone():
                    break

                counter += 1
                username = f"{original_username}-{counter}"

        # Generate secure password
        import secrets
        password = secrets.token_urlsafe(32)

        # Register account using HMAC shared secret (same as supervisor account)
        import hmac
        import hashlib
        import httpx

        # Get nonce first
        register_url = f"{self.homeserver_url}/_synapse/admin/v1/register"
        async with httpx.AsyncClient() as http_client:
            nonce_response = await http_client.get(register_url)
            nonce_response.raise_for_status()
            nonce = nonce_response.json()["nonce"]

            # Compute HMAC
            mac = hmac.new(
                key=self.registration_secret.encode('utf-8'),
                digestmod=hashlib.sha1,
            )
            mac.update(nonce.encode('utf-8'))
            mac.update(b"\x00")
            mac.update(username.encode('utf-8'))
            mac.update(b"\x00")
            mac.update(password.encode('utf-8'))
            mac.update(b"\x00")
            mac.update(b"notadmin")

            # Register via Synapse admin API
            response = await http_client.post(
                register_url,
                json={
                    "nonce": nonce,
                    "username": username,
                    "password": password,
                    "admin": False,
                    "mac": mac.hexdigest(),
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

        matrix_user_id = f"@{username}:{self.homeserver_name}"

        # Login to get access token
        from nio import AsyncClient
        client = AsyncClient(self.homeserver_url, matrix_user_id)

        try:
            login_response = await client.login(password)

            if hasattr(login_response, 'access_token'):
                access_token = login_response.access_token

                # Store credentials in database
                with StateManager(self.state_db_path) as state:
                    state.store_agent_matrix_credentials(
                        agent_id=agent_id,
                        matrix_user_id=matrix_user_id,
                        access_token=access_token
                    )

                return matrix_user_id, access_token
            else:
                raise RuntimeError(f"Failed to login agent account: {login_response}")
        finally:
            await client.close()

    async def deactivate_agent_account(self, agent_id: str):
        """Deactivate an agent's Matrix account (logout)

        Note: Matrix doesn't have a proper 'deactivate' API. This logs out the account
        which invalidates the access token. The account persists and can be reactivated.
        """
        from blue_fairy_supervisor.state import StateManager

        # Get agent's credentials
        with StateManager(self.state_db_path) as state:
            creds = state.get_agent_matrix_credentials(agent_id)

        if not creds:
            # No account to deactivate
            return

        # Logout to invalidate access token
        from nio import AsyncClient
        client = AsyncClient(self.homeserver_url, creds["matrix_user_id"])
        client.access_token = creds["access_token"]

        try:
            await client.logout()
        except Exception as e:
            # Account may already be logged out, that's OK
            pass
        finally:
            await client.close()

        # Keep credentials in database - account is just inactive, not deleted

    async def delete_agent_account(self, agent_id: str):
        """Delete an agent's Matrix account permanently

        This deactivates the account on the homeserver (if possible) and removes
        credentials from database. Used when agent is permanently removed.

        Note: This respects agent identity - accounts are only deleted when the
        agent itself is removed, not when temporarily stopped.
        """
        from blue_fairy_supervisor.state import StateManager

        # Get agent's credentials
        with StateManager(self.state_db_path) as state:
            creds = state.get_agent_matrix_credentials(agent_id)

        if not creds:
            # No account to delete - this is OK, might have been deleted already
            return

        # Try to deactivate account on Synapse using admin API
        # This is best-effort - if it fails, we still remove from our database
        try:
            import httpx
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    f"{self.homeserver_url}/_synapse/admin/v1/deactivate/{creds['matrix_user_id']}",
                    headers={"Authorization": f"Bearer {self.supervisor_access_token}"},
                    json={"erase": True},  # Erase user data
                    timeout=30.0
                )
                # 200 (success) or 404 (already deactivated) are both OK
                if response.status_code not in [200, 404]:
                    print(f"Warning: Synapse deactivation returned {response.status_code}")
        except Exception as e:
            # Log but don't fail - we'll still remove from our database
            print(f"Warning: Failed to deactivate account on homeserver: {e}")

        # Remove credentials from database
        with StateManager(self.state_db_path) as state:
            state.delete_agent_matrix_credentials(agent_id)

    def stop_synapse(self):
        """Stop Synapse container (preserves data)"""
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            container.stop(timeout=10)
        except docker.errors.NotFound:
            pass

    def start_synapse(self):
        """Start stopped Synapse container"""
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            raise Exception("Synapse container not found - call spawn_synapse() first")

    def remove_synapse(self):
        """Remove Synapse container and all data

        WARNING: This removes the container but preserves data in the data directory.
        To completely remove all Synapse data, manually delete the matrix data directory.
        """
        # Stop first
        self.stop_synapse()

        # Remove container
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            container.remove()
        except docker.errors.NotFound:
            pass

        # Data directory is NOT removed for safety
        # Users should manually delete matrix_dir if they want to remove all data

    def is_synapse_running(self) -> bool:
        """Check if Synapse container is running"""
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            container.reload()  # Refresh container status
            return container.status == "running"
        except docker.errors.NotFound:
            return False

    async def create_room(
        self,
        name: str,
        topic: Optional[str] = None,
        invite: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create a new Matrix room

        Args:
            name: Human-readable room name
            topic: Optional room topic/description
            invite: Optional list of Matrix user IDs to invite
            created_by: Matrix user ID who created the room

        Returns:
            (room_id, room_alias) tuple
        """
        # Input validation
        if not name or not name.strip():
            raise ValueError("Room name cannot be empty")

        if not self.supervisor_user_id or not self.supervisor_access_token or not self.homeserver_url:
            raise RuntimeError("Matrix client not initialized. Call create_supervisor_account() first.")

        # Create a fresh client for this event loop
        client = AsyncClient(self.homeserver_url, self.supervisor_user_id)
        client.access_token = self.supervisor_access_token
        client.user_id = self.supervisor_user_id

        try:
            # Generate room alias from name (sanitize: lowercase, replace spaces/special chars with hyphens)
            alias_localpart = re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-'))
            alias_localpart = re.sub(r'-+', '-', alias_localpart).strip('-')  # Remove consecutive/trailing hyphens

            # Handle alias conflicts by appending number if needed
            from blue_fairy_supervisor.state import StateManager

            with StateManager(self.state_db_path) as state:
                original_alias = f"#{alias_localpart}:{self.homeserver_name}"
                room_alias = original_alias
                counter = 2

                # Check if alias already exists
                cursor = state.conn.cursor()
                while True:
                    existing = cursor.execute(
                        "SELECT room_id FROM rooms WHERE room_alias = ?",
                        (room_alias,)
                    ).fetchone()
                    if not existing:
                        break
                    room_alias = f"#{alias_localpart}-{counter}:{self.homeserver_name}"
                    counter += 1

                # Create room via Matrix Client-Server API
                response = await client.room_create(
                    name=name,
                    topic=topic,
                    alias=alias_localpart if counter == 2 else f"{alias_localpart}-{counter-1}",
                    invite=invite or [],
                    visibility=RoomVisibility.private,  # Private rooms by default
                    preset=RoomPreset.private_chat  # Standard power levels
                )

                if hasattr(response, 'room_id'):
                    room_id = response.room_id

                    # Store in database
                    state.add_room(
                        room_id=room_id,
                        room_name=name,
                        room_alias=room_alias,
                        topic=topic,
                        created_by=created_by or self.supervisor_user_id
                    )

                    # Add supervisor as member
                    state.add_room_member(
                        room_id=room_id,
                        user_id=self.supervisor_user_id,
                        member_type="supervisor"
                    )

                    return room_id, room_alias
                else:
                    raise RuntimeError(f"Failed to create room: {response}")
        finally:
            await client.close()

    def _resolve_room_identifier(self, room_identifier: str) -> str:
        """Resolve room alias to room ID if needed.

        Args:
            room_identifier: Room ID (!abc:domain) or alias (#name:domain)

        Returns:
            Room ID

        Raises:
            ValueError: If alias not found in database
        """
        if room_identifier.startswith("#"):
            # Look up room ID from alias in database
            from blue_fairy_supervisor.state import StateManager

            with StateManager(self.state_db_path) as state:
                result = state.conn.execute(
                    "SELECT room_id FROM rooms WHERE room_alias = ?",
                    (room_identifier,)
                ).fetchone()

            if not result:
                raise ValueError(f"Room alias '{room_identifier}' not found")

            return result[0]

        # Already a room ID
        return room_identifier

    async def delete_room(self, room_id: str):
        """Archive a room (Matrix doesn't support true deletion)

        This kicks all members and marks the room as archived.

        Args:
            room_id: The Matrix room ID (!abc:domain) or alias (#name:domain) to delete/archive
        """
        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        if not self.supervisor_user_id or not self.supervisor_access_token or not self.homeserver_url:
            raise RuntimeError("Matrix client not initialized")

        # Create a fresh client for this event loop
        client = AsyncClient(self.homeserver_url, self.supervisor_user_id)
        client.access_token = self.supervisor_access_token
        client.user_id = self.supervisor_user_id

        try:
            from blue_fairy_supervisor.state import StateManager

            with StateManager(self.state_db_path) as state:
                # Get all current members
                members = state.get_room_members(room_id, include_left=False)

                # Kick all members except supervisor
                for member in members:
                    if member["user_id"] != self.supervisor_user_id:
                        await client.room_kick(
                            room_id=room_id,
                            user_id=member["user_id"],
                            reason="Room archived"
                        )
                        state.remove_room_member(room_id, member["user_id"])

                # Supervisor leaves last
                await client.room_leave(room_id)
                state.remove_room_member(room_id, self.supervisor_user_id)

                # Mark room as archived
                state.archive_room(room_id)
        finally:
            await client.close()

    async def invite_to_room(self, room_id: str, user_id: str):
        """Invite a user to a room

        Args:
            room_id: The Matrix room ID (!abc:domain) or alias (#name:domain)
            user_id: The Matrix user ID to invite
        """
        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        if not self.supervisor_user_id or not self.supervisor_access_token or not self.homeserver_url:
            raise RuntimeError("Matrix client not initialized")

        # Create a fresh client for this event loop
        client = AsyncClient(self.homeserver_url, self.supervisor_user_id)
        client.access_token = self.supervisor_access_token
        client.user_id = self.supervisor_user_id

        try:
            # Invite via Matrix API
            response = await client.room_invite(room_id, user_id)

            # Check if invite was successful (nio returns RoomInviteError on failure)
            from nio import RoomInviteError
            if isinstance(response, RoomInviteError):
                raise RuntimeError(f"Failed to invite user: {response}")

            # Add to database (they may not have joined yet, but we track the invite)
            from blue_fairy_supervisor.state import StateManager

            with StateManager(self.state_db_path) as state:
                # Determine member type (external for now; Phase 3 will add agent support)
                member_type = "external"

                state.add_room_member(
                    room_id=room_id,
                    user_id=user_id,
                    member_type=member_type
                )
        finally:
            await client.close()

    async def invite_agent_to_room(self, room_id: str, agent_id: str):
        """Invite an agent to a room by agent_id

        This looks up the agent's Matrix user ID and invites them to the room.
        The database tracks this as an agent member (not human).

        Args:
            room_id: Matrix room ID (!abc:domain) or alias (#name:domain)
            agent_id: Agent identifier (e.g., "researcher-a8f2")

        Raises:
            ValueError: If no Matrix account exists for this agent
            RuntimeError: If the invite fails
        """
        from blue_fairy_supervisor.state import StateManager
        from nio import AsyncClient, RoomInviteError

        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        # Get agent's Matrix credentials
        with StateManager(self.state_db_path) as state:
            creds = state.get_agent_matrix_credentials(agent_id)

        if not creds:
            raise ValueError(f"No Matrix account found for agent {agent_id}")

        matrix_user_id = creds["matrix_user_id"]

        # Invite via Matrix API using supervisor's credentials
        client = AsyncClient(self.homeserver_url, self.supervisor_user_id)
        client.access_token = self.supervisor_access_token

        try:
            response = await client.room_invite(room_id, matrix_user_id)

            if isinstance(response, RoomInviteError):
                raise RuntimeError(f"Failed to invite agent {agent_id}: {response.message}")

            # Add to database with agent_id tracked (so we know it's an agent)
            with StateManager(self.state_db_path) as state:
                state.add_room_member(
                    room_id=room_id,
                    user_id=matrix_user_id,
                    member_type="agent",
                    agent_id=agent_id
                )
        finally:
            await client.close()

    async def remove_from_room(self, room_id: str, user_id: str):
        """Remove a user from a room (kick)

        Args:
            room_id: The Matrix room ID (!abc:domain) or alias (#name:domain)
            user_id: The Matrix user ID to remove
        """
        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        if not self.supervisor_user_id or not self.supervisor_access_token or not self.homeserver_url:
            raise RuntimeError("Matrix client not initialized")

        # Create a fresh client for this event loop
        client = AsyncClient(self.homeserver_url, self.supervisor_user_id)
        client.access_token = self.supervisor_access_token
        client.user_id = self.supervisor_user_id

        try:
            # Kick via Matrix API
            await client.room_kick(room_id, user_id)

            # Update database
            from blue_fairy_supervisor.state import StateManager

            with StateManager(self.state_db_path) as state:
                state.remove_room_member(room_id, user_id)
        finally:
            await client.close()

    def list_rooms(self, include_archived: bool = False) -> List[dict]:
        """List all rooms from state database

        Args:
            include_archived: Whether to include archived rooms

        Returns:
            List of room dictionaries
        """
        from blue_fairy_supervisor.state import StateManager

        with StateManager(self.state_db_path) as state:
            rooms = state.list_rooms(include_archived=include_archived)
        return rooms

    def get_room_members(self, room_id: str) -> List[dict]:
        """Get all current members of a room

        Args:
            room_id: The Matrix room ID (!abc:domain) or alias (#name:domain)

        Returns:
            List of member dictionaries
        """
        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        from blue_fairy_supervisor.state import StateManager

        with StateManager(self.state_db_path) as state:
            members = state.get_room_members(room_id, include_left=False)
        return members

    def get_agent_account(self, agent_id: str) -> dict | None:
        """Get Matrix account for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with matrix_user_id and created_at, or None if not found
        """
        from blue_fairy_supervisor.state import StateManager

        with StateManager(self.state_db_path) as state:
            result = state.conn.execute(
                "SELECT matrix_user_id, created_at FROM matrix_accounts WHERE agent_id = ?",
                (agent_id,)
            ).fetchone()

        if not result:
            return None

        return {
            "matrix_user_id": result[0],
            "created_at": result[1]
        }

    def is_synapse_running(self) -> bool:
        """Check if Synapse container is running.

        Returns:
            True if Synapse container exists and is running
        """
        try:
            container = self.docker.containers.get("blue-fairy-synapse")
            return container.status == "running"
        except docker.errors.NotFound:
            return False

    async def send_room_message(self, room_id: str, message: str, as_user: str = "supervisor") -> dict:
        """Send a message to a Matrix room.

        Args:
            room_id: Matrix room ID (e.g., "!abc:domain") or alias (e.g., "#name:domain")
            message: Message content to send
            as_user: Which user to send as - "supervisor" (default) or agent_id

        Returns:
            Dict with event_id and timestamp

        Raises:
            ValueError: If room not found or user doesn't have access
            RuntimeError: If send fails
        """
        from blue_fairy_supervisor.state import StateManager

        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        # Get credentials for the user sending the message
        if as_user == "supervisor":
            # Use supervisor credentials
            with StateManager(self.state_db_path) as state:
                result = state.conn.execute(
                    "SELECT user_id, access_token FROM matrix_supervisor WHERE id = 1"
                ).fetchone()

            if not result:
                raise RuntimeError("Supervisor account not found")

            user_id, access_token = result
        else:
            # Use agent credentials
            with StateManager(self.state_db_path) as state:
                result = state.conn.execute(
                    "SELECT matrix_user_id, access_token FROM matrix_accounts WHERE agent_id = ?",
                    (as_user,)
                ).fetchone()

            if not result:
                raise ValueError(f"Agent '{as_user}' Matrix account not found")

            user_id, access_token = result

        # Get homeserver URL
        homeserver_url = self.get_homeserver_url()

        # Create Matrix client
        client = AsyncClient(homeserver_url, user_id)
        client.access_token = access_token

        try:
            # Send message
            response = await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message
                }
            )

            if hasattr(response, "event_id"):
                return {
                    "event_id": response.event_id,
                    "room_id": room_id,
                    "sender": user_id
                }
            else:
                raise RuntimeError(f"Failed to send message: {response}")

        finally:
            await client.close()

    async def get_room_messages(
        self,
        room_id: str,
        limit: int = 50,
        as_user: str = "supervisor"
    ) -> list[dict]:
        """Get message history from a Matrix room.

        Args:
            room_id: Matrix room ID (e.g., "!abc:domain") or alias (e.g., "#name:domain")
            limit: Maximum number of messages to retrieve (default: 50)
            as_user: Which user to retrieve as - "supervisor" (default) or agent_id

        Returns:
            List of message dicts with sender, content, timestamp, event_id

        Raises:
            ValueError: If room not found or user doesn't have access
        """
        from blue_fairy_supervisor.state import StateManager

        # Resolve alias to room ID if needed
        room_id = self._resolve_room_identifier(room_id)

        # Get credentials
        if as_user == "supervisor":
            with StateManager(self.state_db_path) as state:
                result = state.conn.execute(
                    "SELECT user_id, access_token FROM matrix_supervisor WHERE id = 1"
                ).fetchone()

            if not result:
                raise RuntimeError("Supervisor account not found")

            user_id, access_token = result
        else:
            with StateManager(self.state_db_path) as state:
                result = state.conn.execute(
                    "SELECT matrix_user_id, access_token FROM matrix_accounts WHERE agent_id = ?",
                    (as_user,)
                ).fetchone()

            if not result:
                raise ValueError(f"Agent '{as_user}' Matrix account not found")

            user_id, access_token = result

        # Get homeserver URL
        homeserver_url = self.get_homeserver_url()

        # Create Matrix client
        client = AsyncClient(homeserver_url, user_id)
        client.access_token = access_token

        try:
            # Sync to get room state
            await client.sync(timeout=0)

            # Get room messages
            response = await client.room_messages(
                room_id=room_id,
                start="",
                limit=limit
            )

            # Check for error response
            if hasattr(response, 'status_code') or not hasattr(response, 'chunk'):
                error_msg = getattr(response, 'message', str(response))
                raise ValueError(f"Failed to get room messages: {error_msg}")

            messages = []
            for event in response.chunk:
                # Only include m.room.message events (RoomMessageText, etc.)
                if hasattr(event, "body"):
                    messages.append({
                        "event_id": event.event_id,
                        "sender": event.sender,
                        "content": event.body,
                        "timestamp": event.server_timestamp,
                        "type": getattr(event, "msgtype", "m.text")
                    })

            # Reverse to get chronological order (oldest first)
            messages.reverse()

            return messages

        finally:
            await client.close()

"""JSON-RPC API for supervisor"""

from typing import Any, Optional, Dict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from nio import AsyncClient

from .state import StateManager
from .matrix_manager import MatrixManager
from blue_fairy_common.types import AgentRuntimeConfig


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request"""
    jsonrpc: str = Field(..., pattern="^2\\.0$")
    method: str
    params: dict = Field(default_factory=dict)
    id: int | str


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[dict] = None
    id: Optional[int | str] = None


class TelemetryEvent(BaseModel):
    """Telemetry event from Level 2 agent"""
    agent_id: str
    event_id: str
    event_type: str  # 'decision', 'memory_retrieval', 'communication', 'reasoning'
    data: Dict[str, Any]


class QueuePushRequest(BaseModel):
    """Request to push event to agent queue"""
    source: str
    summary: str


class SupervisorAPI:
    """Supervisor JSON-RPC API handlers"""

    def __init__(
        self,
        state: StateManager,
        lifecycle: 'LifecycleManager'
    ):
        self.state = state
        self.lifecycle = lifecycle

    def list_agents(self, params: dict) -> list[dict]:
        """List all agents"""
        agents = self.state.list_agents()
        return [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "template": a.template,
                "status": a.status,
                "public_key": a.public_key,
                "started_at": a.created_at.isoformat(),
            }
            for a in agents
        ]

    async def spawn_agent(self, params: dict) -> dict:
        """Spawn an agent from template"""
        template = params.get("template")
        agent_id = params.get("agent_id")

        if not template:
            raise ValueError("template is required")

        return await self.lifecycle.spawn_agent(template, agent_id)

    def send_message(self, params: dict) -> dict:
        """Send message to agent"""
        agent_id = params.get("agent_id")
        message = params.get("message")

        if not agent_id or not message:
            raise ValueError("agent_id and message are required")

        return self.lifecycle.send_message(agent_id, message)

    def stop_agent(self, params: dict) -> dict:
        """Stop an agent"""
        agent_id = params.get("agent_id")

        if not agent_id:
            raise ValueError("agent_id is required")

        return self.lifecycle.stop_agent(agent_id)

    async def start_agent(self, params: dict) -> dict:
        """Start a stopped agent"""
        agent_id = params.get("agent_id")

        if not agent_id:
            raise ValueError("agent_id is required")

        return await self.lifecycle.start_agent(agent_id)

    async def remove_agent(self, params: dict) -> dict:
        """Remove an agent completely"""
        agent_id = params.get("agent_id")

        if not agent_id:
            raise ValueError("agent_id is required")

        return await self.lifecycle.remove_agent(agent_id)

    async def upgrade_agent(self, params: dict) -> dict:
        """Upgrade an agent's container"""
        agent_id = params.get("agent_id")
        rebuild = params.get("rebuild", False)
        refresh_config = params.get("refresh_config", False)

        if not agent_id:
            raise ValueError("agent_id is required")

        return await self.lifecycle.upgrade_agent(
            agent_id,
            rebuild=rebuild,
            refresh_config=refresh_config
        )


def create_app(
    db_path: Path | str,
    config_path: Optional[Path | str] = None
) -> FastAPI:
    """Create FastAPI app with JSON-RPC endpoint"""
    from blue_fairy_common import ConfigLoader
    from .docker_mgr import DockerManager
    from .lifecycle import LifecycleManager
    import time

    app = FastAPI(title="Blue Fairy Supervisor")

    # Load configuration
    if config_path:
        config_loader = ConfigLoader()
        config = config_loader.load(config_path)
    else:
        # Use default config location
        default_config = Path.home() / ".blue-fairy" / "config.yaml"
        if default_config.exists():
            config_loader = ConfigLoader()
            config = config_loader.load(default_config)
        else:
            raise ValueError("No config file provided and default not found")

    # Initialize components
    state = StateManager(db_path)
    docker = DockerManager()
    matrix = MatrixManager(state_db_path=db_path)
    lifecycle = LifecycleManager(state, docker, config, matrix_manager=matrix)
    supervisor = SupervisorAPI(state, lifecycle)

    # Background task for Matrix initialization
    async def initialize_matrix_background():
        """Initialize Matrix homeserver and supervisor account in background"""
        import asyncio
        import sys
        import traceback
        import docker as docker_sdk

        try:
            print("[MATRIX-INIT] Starting Matrix initialization background task")

            # Try to load existing supervisor credentials first
            print("[MATRIX-INIT] Checking for existing credentials...")
            if matrix.load_supervisor_credentials():
                print("✓ Loaded existing supervisor credentials from database")
            else:
                print("[MATRIX-INIT] No existing credentials found")

            # Generate Synapse config if needed
            print("[MATRIX-INIT] Checking registration secret...")
            if not matrix.registration_secret:
                print("[MATRIX-INIT] Generating Synapse config...")
                matrix.generate_synapse_config()
            else:
                print("[MATRIX-INIT] Registration secret already exists")

            # Spawn Synapse if not running
            print("[MATRIX-INIT] Checking if Synapse is running...")
            if not matrix.is_synapse_running():
                print("[MATRIX-INIT] Spawning Synapse container...")
                matrix.spawn_synapse()

                # Wait for Synapse container to be healthy (check status every 2s for up to 120s)
                print("[MATRIX-INIT] Waiting for Synapse to become healthy...")
                max_wait = 120
                for i in range(max_wait // 2):
                    try:
                        container = docker_sdk.from_env().containers.get("blue-fairy-synapse")
                        container.reload()
                        health = container.attrs.get('State', {}).get('Health', {}).get('Status')
                        if health == 'healthy':
                            print(f"✓ Synapse healthy after {(i+1)*2}s")
                            break
                    except Exception as e:
                        print(f"[MATRIX-INIT] Health check error (attempt {i+1}): {e}")
                    await asyncio.sleep(2)
                else:
                    print(f"Warning: Synapse not healthy after {max_wait}s, proceeding anyway")
            else:
                print("[MATRIX-INIT] Synapse already running")

            # Get homeserver URL for client creation
            print("[MATRIX-INIT] Getting homeserver URL...")
            try:
                container = docker_sdk.from_env().containers.get("blue-fairy-synapse")
                container.reload()
                port_bindings = container.attrs['NetworkSettings']['Ports'].get(f'{matrix.SYNAPSE_PORT}/tcp')
                if port_bindings and port_bindings[0]:
                    host_port = port_bindings[0]['HostPort']
                    matrix.homeserver_url = f"http://localhost:{host_port}"
                    print(f"[MATRIX-INIT] Homeserver URL: {matrix.homeserver_url}")
            except Exception as e:
                print(f"[MATRIX-INIT] Failed to get homeserver URL: {e}")

            # Create or verify supervisor account
            print("[MATRIX-INIT] Checking supervisor account...")
            if not matrix.supervisor_user_id:
                # No credentials loaded, create new account
                print("[MATRIX-INIT] Creating supervisor account...")
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"[MATRIX-INIT] Account creation attempt {attempt + 1}/{max_retries}")
                        username, password = await matrix.create_supervisor_account_async()
                        print(f"✓ Supervisor account created: {username}")
                        break
                    except Exception as e:
                        print(f"[MATRIX-INIT] Account creation failed: {type(e).__name__}: {e}")
                        if attempt < max_retries - 1:
                            print(f"Retry {attempt + 1}/{max_retries}: Supervisor account creation failed, waiting...")
                            await asyncio.sleep(5)
                        else:
                            print(f"Warning: Failed to create supervisor account after {max_retries} attempts: {e}")
                            traceback.print_exc()
            else:
                # Credentials loaded
                print(f"✓ Using existing supervisor account: {matrix.supervisor_user_id}")

            # Create Matrix client if we have credentials and homeserver URL
            print("[MATRIX-INIT] Initializing Matrix client...")
            if matrix.supervisor_user_id and matrix.homeserver_url:
                try:
                    matrix.client = AsyncClient(matrix.homeserver_url, matrix.supervisor_user_id)
                    matrix.client.access_token = matrix.supervisor_access_token
                    print("✓ Matrix client initialized")
                except Exception as e:
                    print(f"Warning: Failed to initialize Matrix client: {e}")
                    traceback.print_exc()
            else:
                print(f"[MATRIX-INIT] Cannot initialize client - user_id: {matrix.supervisor_user_id}, url: {matrix.homeserver_url}")

            print("[MATRIX-INIT] Background task completed")

        except Exception as e:
            print(f"[MATRIX-INIT] FATAL ERROR in background task: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    # Start Matrix initialization in background
    @app.on_event("startup")
    async def start_matrix_init():
        """Start Matrix initialization as background task"""
        import asyncio
        from .observability import ObservabilityManager

        # Initialize observability manager
        state_dir = Path(db_path).parent
        obs_db_path = state_dir / "observability.db"
        app.state.observability_manager = ObservabilityManager(obs_db_path)

        asyncio.create_task(initialize_matrix_background())

    # Method registry
    methods = {
        "list_agents": supervisor.list_agents,
        "spawn_agent": supervisor.spawn_agent,
        "send_message": supervisor.send_message,
        "stop_agent": supervisor.stop_agent,
        "start_agent": supervisor.start_agent,
        "remove_agent": supervisor.remove_agent,
        "upgrade_agent": supervisor.upgrade_agent,
    }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "ok"}

    @app.post("/rpc")
    async def rpc_endpoint(request: Request):
        """JSON-RPC 2.0 endpoint"""
        import asyncio
        import inspect

        try:
            body = await request.json()
            rpc_request = JSONRPCRequest(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        # Look up method
        if rpc_request.method not in methods:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                },
                "id": rpc_request.id
            })

        # Execute method
        try:
            method = methods[rpc_request.method]

            # Check if method is async
            if inspect.iscoroutinefunction(method):
                # Call async method directly
                result = await method(rpc_request.params)
            else:
                # Run synchronous method in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(method, rpc_request.params)

            return JSONResponse({
                "jsonrpc": "2.0",
                "result": result,
                "id": rpc_request.id
            })
        except NotImplementedError as e:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": rpc_request.id
            })
        except Exception as e:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {e}"
                },
                "id": rpc_request.id
            })

    # Room management endpoints

    @app.post("/rooms/create")
    async def create_room(request: Request):
        """Create a new chatroom"""
        body = await request.json()
        name = body.get("name")
        topic = body.get("topic")
        invite = body.get("invite", [])
        created_by = body.get("created_by")

        if not name:
            return JSONResponse({"error": "Room name is required"}, status_code=400)

        matrix = app.state.matrix_manager

        # Matrix should be initialized at startup, but check anyway
        if not matrix.client:
            return JSONResponse(
                {"error": "Matrix homeserver not ready. Please try again in a few seconds."},
                status_code=503
            )

        try:
            room_id, room_alias = await matrix.create_room(
                name=name,
                topic=topic,
                invite=invite,
                created_by=created_by
            )
            return JSONResponse({"room_id": room_id, "room_alias": room_alias})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/rooms")
    async def list_rooms(include_archived: bool = False):
        """List all rooms"""
        matrix = app.state.matrix_manager
        rooms = matrix.list_rooms(include_archived=include_archived)
        return JSONResponse({"rooms": rooms})

    @app.post("/rooms/delete")
    async def delete_room(request: Request):
        """Archive a room"""
        body = await request.json()
        room_id = body.get("room_id")

        if not room_id:
            return JSONResponse({"error": "room_id is required"}, status_code=400)

        matrix = app.state.matrix_manager

        try:
            await matrix.delete_room(room_id)
            return JSONResponse({"status": "deleted", "room_id": room_id})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/rooms/invite")
    async def invite_to_room(request: Request):
        """Invite user or agent to room

        Request body must contain:
        - room_id: Matrix room ID (e.g., "!abc123:blue-fairy.local")

        And either:
        - user_id: Matrix user ID (e.g., "@alice:blue-fairy.local")
        - agent_id: Agent identifier (e.g., "researcher-a8f2")
        """
        import logging
        logger = logging.getLogger("uvicorn")
        logger.info("[INVITE] Endpoint called!")
        body = await request.json()
        logger.info(f"[INVITE] Request body: {body}")
        room_id = body.get("room_id")
        user_id = body.get("user_id")
        agent_id = body.get("agent_id")

        if not room_id:
            return JSONResponse({"error": "room_id is required"}, status_code=400)

        if not user_id and not agent_id:
            return JSONResponse({"error": "Either user_id or agent_id is required"}, status_code=400)

        if user_id and agent_id:
            return JSONResponse({"error": "Provide only one of user_id or agent_id, not both"}, status_code=400)

        matrix = app.state.matrix_manager

        try:
            if agent_id:
                # Invite by agent_id (new functionality)
                await matrix.invite_agent_to_room(room_id, agent_id)
                return JSONResponse({
                    "status": "invited",
                    "agent_id": agent_id,
                    "room_id": room_id
                })
            else:
                # Invite by matrix user_id (existing functionality)
                await matrix.invite_to_room(room_id, user_id)
                return JSONResponse({
                    "status": "invited",
                    "user_id": user_id,
                    "room_id": room_id
                })
        except ValueError as e:
            logger.error(f"[INVITE] ValueError: {e}")
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            logger.error(f"[INVITE] Exception: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/rooms/remove")
    async def remove_from_room(request: Request):
        """Remove user from room"""
        body = await request.json()
        room_id = body.get("room_id")
        user_id = body.get("user_id")

        if not room_id:
            return JSONResponse({"error": "room_id is required"}, status_code=400)

        if not user_id:
            return JSONResponse({"error": "user_id is required"}, status_code=400)

        matrix = app.state.matrix_manager

        try:
            await matrix.remove_from_room(room_id, user_id)
            return JSONResponse({"status": "removed", "user_id": user_id, "room_id": room_id})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/rooms/members")
    async def get_room_members(request: Request):
        """Get room members"""
        body = await request.json()
        room_id = body.get("room_id")

        if not room_id:
            return JSONResponse({"error": "room_id is required"}, status_code=400)

        matrix = app.state.matrix_manager
        members = matrix.get_room_members(room_id)
        return JSONResponse({"members": members})

    @app.post("/rooms/send")
    async def send_room_message(request: Request):
        """Send a message to a room"""
        body = await request.json()
        room_id = body.get("room_id")
        message = body.get("message")
        as_user = body.get("as_user", "supervisor")

        if not room_id or not message:
            return JSONResponse({"error": "room_id and message are required"}, status_code=400)

        try:
            matrix = app.state.matrix_manager
            result = await matrix.send_room_message(room_id, message, as_user=as_user)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # Telemetry endpoint
    @app.post("/telemetry/event")
    async def receive_telemetry_event(event: TelemetryEvent):
        """Receive and store telemetry event from Level 2 agent

        Args:
            event: Telemetry event payload

        Returns:
            Status confirmation
        """
        try:
            # Get observability manager (initialized in app state)
            obs_manager = app.state.observability_manager

            obs_manager.store_event(
                agent_id=event.agent_id,
                event_id=event.event_id,
                event_type=event.event_type,
                data=event.data
            )

            return {"status": "stored", "event_id": event.event_id}

        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    # Queue management endpoints
    @app.post("/agents/{agent_id}/queue")
    async def push_queue_item(agent_id: str, request: QueuePushRequest):
        """Push an event to agent's queue.

        Used by plugins/listeners to queue events for agent processing.
        """
        agent = state.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        item_id = state.push_queue_item(
            agent_id=agent_id,
            source=request.source,
            summary=request.summary
        )

        return {"id": item_id, "status": "queued"}

    # ==================== REST API for Web UI ====================

    @app.get("/agents")
    async def get_agents():
        """List all agents with status (REST endpoint for Web UI)"""
        agents = state.list_agents()
        # Convert AgentState objects to dicts for JSON serialization
        agents_data = [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "template": a.template,
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in agents
        ]
        return JSONResponse({"agents": agents_data})

    @app.get("/agents/{agent_id}/config")
    async def get_agent_config(agent_id: str):
        """Return runtime config for agent harness.

        This endpoint is called by the harness on container startup
        to fetch its configuration instead of reading from a mounted file.

        Note: Requires spawn_agent to store prompt_text, memory_enabled, and
        memory_backend in agent.config (implemented in Task 9 of refactor plan).
        """
        agent = state.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        return AgentRuntimeConfig(
            agent_id=agent.agent_id,
            model=agent.config.get("model", ""),
            prompt_text=agent.config.get("prompt_text", ""),
            memory_enabled=agent.config.get("memory_enabled", False),
            memory_backend=agent.config.get("memory_backend", "mem0"),
            remem_enabled=agent.config.get("remem_enabled", False),
            action_quota_limit=agent.config.get("action_quota_limit"),
            action_quota_window=agent.config.get("action_quota_window"),
            action_quota_low_threshold=agent.config.get("action_quota_low_threshold", 0.3),
            subroutines_enabled=agent.config.get("subroutines_enabled", False),
            subroutines_config=agent.config.get("subroutines_config"),
            mcp_servers=agent.config.get("mcp_servers"),
            plugins=agent.config.get("plugins", []),
        )

    @app.get("/timeline/{agent_id}")
    async def get_timeline(agent_id: str, event_types: str = None, limit: int = 100):
        """Get agent timeline (REST endpoint for Web UI)

        Args:
            agent_id: Agent identifier
            event_types: Comma-separated event types to filter (e.g., "decision,communication")
            limit: Maximum events to return (default 100)
        """
        obs_manager = app.state.observability_manager
        if not obs_manager:
            return JSONResponse({"error": "Observability not available"}, status_code=503)

        # Parse event types filter
        types_filter = None
        if event_types:
            types_filter = [t.strip() for t in event_types.split(",")]

        timeline = obs_manager.get_timeline(
            agent_id=agent_id,
            event_types=types_filter
        )

        # Apply limit
        if limit and len(timeline) > limit:
            timeline = timeline[:limit]

        return JSONResponse({"timeline": timeline, "count": len(timeline)})

    @app.get("/rooms/{room_id:path}/messages")
    async def get_room_messages_rest(room_id: str, limit: int = 50):
        """Get room messages (REST endpoint for Web UI)

        Args:
            room_id: Room ID or alias (URL-encoded)
            limit: Maximum messages to return (default 50)
        """
        import urllib.parse
        room_id = urllib.parse.unquote(room_id)

        matrix = app.state.matrix_manager
        if not matrix or not matrix.homeserver_url:
            return JSONResponse({"error": "Matrix not available"}, status_code=503)

        try:
            messages = await matrix.get_room_messages(room_id, limit=limit)
            return JSONResponse({"messages": messages, "count": len(messages)})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ==================== Static File Serving ====================

    from fastapi.responses import FileResponse

    static_dir = Path(__file__).parent / "static"

    @app.get("/ui")
    async def serve_ui():
        """Serve the Web UI"""
        index_path = static_dir / "index.html"
        if not index_path.exists():
            return JSONResponse({"error": "Web UI not found"}, status_code=404)
        return FileResponse(index_path, media_type="text/html")

    # Store managers for cleanup
    app.state.state_manager = state
    app.state.matrix_manager = matrix

    # Debug: Log all registered routes
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("[DEBUG] Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"  {route.path}: {route.methods}")

    return app


def create_app_from_env() -> FastAPI:
    """Create FastAPI app from environment variables

    Reads:
    - DB_PATH: Path to SQLite database
    - CONFIG_PATH: Path to config YAML (optional, uses default)
    """
    import os

    db_path = os.getenv("DB_PATH")
    if not db_path:
        # Use default
        db_path = Path.home() / ".blue-fairy" / "state.db"

    config_path = os.getenv("CONFIG_PATH")

    return create_app(db_path=db_path, config_path=config_path)

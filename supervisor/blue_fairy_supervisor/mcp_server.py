"""MCP server for Blue Fairy supervisor inspection tools."""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import logging
import json
import requests
import time
import base64
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .otel_client import OTelClient

logger = logging.getLogger(__name__)


class BlueFairyMCPServer:
    """MCP server providing inspection tools for Blue Fairy infrastructure.

    Privacy Boundary: Only exposes infrastructure state (containers, accounts, rooms).
    Does NOT expose agent memory, reasoning, or internal thoughts.

    Observability tools (Phase 5) respect agent consent and only show data
    for agents with observability level >= 1.
    """

    def __init__(self, state_manager, matrix_manager, docker_manager, observability_manager=None, lifecycle_manager=None):
        """Initialize MCP server with supervisor components.

        Args:
            state_manager: StateManager instance for database access
            matrix_manager: MatrixManager instance for Matrix operations
            docker_manager: DockerManager instance for container inspection
            observability_manager: ObservabilityManager instance for telemetry data (optional)
            lifecycle_manager: LifecycleManager instance for agent lifecycle operations (optional)
        """
        self.state_manager = state_manager
        self.matrix_manager = matrix_manager
        self.docker_manager = docker_manager
        self.observability_manager = observability_manager
        self.lifecycle_manager = lifecycle_manager
        self.server = Server("blue-fairy-supervisor")

        # Register tool handlers
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="list_agents",
                    description="List all agents with their current status, configuration, and metadata. Returns agent_id, name, template, status, container_id, and creation timestamp.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_agent_details",
                    description="Get detailed information about a specific agent including status, configuration, container stats, and Matrix account.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "The agent ID to get details for"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="list_rooms",
                    description="List all Matrix rooms managed by Blue Fairy. Returns room_id, name, alias, topic, creator, and archived status.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_archived": {
                                "type": "boolean",
                                "description": "Include archived rooms in results (default: false)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_room_members",
                    description="Get members of a specific Matrix room. Accepts room_id or room_alias. Returns user_id, member_type (agent/human), agent_id (if applicable), join/leave timestamps.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "room_identifier": {
                                "type": "string",
                                "description": "Room ID (!abc:domain) or alias (#name:domain)"
                            }
                        },
                        "required": ["room_identifier"]
                    }
                ),
                Tool(
                    name="supervisor_health",
                    description="Get overall health status of Blue Fairy supervisor, including agent counts, Matrix status, and Docker network health.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_agent_timeline",
                    description="Get chronological timeline of events and metrics for an agent. Returns ordered list of events and metrics with timestamps, types, and data. Useful for understanding agent development and behavior over time. Only works for agents with observability level >= 1.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Start timestamp in ISO format (e.g., '2025-11-11T10:00:00'), optional"
                            },
                            "end_time": {
                                "type": "string",
                                "description": "End timestamp in ISO format, optional"
                            },
                            "event_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by event types (e.g., ['decision', 'memory_retrieval']), optional"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="get_agent_metrics",
                    description="Get aggregated metrics for an agent (message counts, response rates, average response time). Returns stats like count, sum, avg, min, max for each metric type. Only works for agents with observability level >= 1.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            },
                            "metric_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by metric types (e.g., ['message_sent', 'response_time']), optional"
                            },
                            "time_range_hours": {
                                "type": "integer",
                                "description": "Time range in hours (default: 24)"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="search_decisions",
                    description="Search agent decision events with filters. Find decisions by agent, type (respond/silent), or text search. Includes reasoning for each decision. Only works for agents with observability level >= 2.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Filter by agent identifier, optional"
                            },
                            "decision_type": {
                                "type": "string",
                                "description": "Filter by decision type ('respond' or 'silent'), optional"
                            },
                            "text_query": {
                                "type": "string",
                                "description": "Search in reasoning/message text, optional"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="send_room_message",
                    description="Send a message to a Matrix room. Useful for experimental interventions, sending prompts to agents, or human participation in rooms. Supports both room IDs and aliases.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "room_id": {
                                "type": "string",
                                "description": "Room ID (!abc:domain) or alias (#name:domain)"
                            },
                            "message": {
                                "type": "string",
                                "description": "Message content to send"
                            },
                            "as_user": {
                                "type": "string",
                                "description": "User to send as: 'supervisor' (default) or agent_id"
                            }
                        },
                        "required": ["room_id", "message"]
                    }
                ),
                Tool(
                    name="get_room_messages",
                    description="Get message history from a Matrix room. Returns structured conversation data with sender, content, and timestamps. Much cleaner than parsing agent logs.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "room_id": {
                                "type": "string",
                                "description": "Room ID (!abc:domain) or alias (#name:domain)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of messages to retrieve (default: 50)"
                            },
                            "as_user": {
                                "type": "string",
                                "description": "User to retrieve as: 'supervisor' (default) or agent_id"
                            }
                        },
                        "required": ["room_id"]
                    }
                ),
                # Subroutine tools
                Tool(
                    name="run_agent_subroutine",
                    description="Run a subroutine on an agent. Subroutines are internal cognitive processes like memory consolidation or dreaming. Returns a task_id for status checking.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            },
                            "subroutine_name": {
                                "type": "string",
                                "description": "Subroutine name (e.g., 'consolidate_memory', 'dream')"
                            },
                            "params": {
                                "type": "object",
                                "description": "Optional parameters for the subroutine"
                            }
                        },
                        "required": ["agent_id", "subroutine_name"]
                    }
                ),
                Tool(
                    name="check_agent_subroutine",
                    description="Check the status of a running or completed subroutine on an agent. Returns status, result, and timestamps.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "Task ID returned from run_agent_subroutine"
                            }
                        },
                        "required": ["agent_id", "task_id"]
                    }
                ),
                Tool(
                    name="list_agent_subroutines",
                    description="List available subroutines for an agent. Shows what internal cognitive processes the agent can run.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="get_agent_sleep_status",
                    description="Get the current sleep state of an agent. Shows whether agent is awake, sleeping, or waking, plus queued message count.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="get_self_version",
                    description="Get this agent's current version and upgrade history.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "The agent ID to get version for"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="request_upgrade",
                    description="Request supervisor to upgrade this agent to a specific commit. Agent will go offline during upgrade.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "The agent ID requesting upgrade"
                            },
                            "commit_sha": {
                                "type": "string",
                                "description": "Git commit SHA to upgrade to"
                            }
                        },
                        "required": ["agent_id", "commit_sha"]
                    }
                ),
                Tool(
                    name="queue_summary",
                    description="Get summary of agent's event queue. Returns total count and counts by source. Used by agents during wake cycles.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="queue_read",
                    description="Read and remove items from agent's event queue. Items are popped (destructive read). Requires signature for authenticated access. Used by agents to consume events during wake cycles.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier"
                            },
                            "signature": {
                                "type": "string",
                                "description": "Authentication signature (timestamp:base64_signature)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum items to return (default: all)"
                            },
                            "oldest_first": {
                                "type": "boolean",
                                "description": "Return oldest items first (default: true)"
                            },
                            "source": {
                                "type": "string",
                                "description": "Filter by source (e.g., 'chat.matrix')"
                            }
                        },
                        "required": ["agent_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            if name == "list_agents":
                return await self._handle_list_agents(arguments)
            elif name == "get_agent_details":
                return await self._handle_get_agent_details(arguments)
            elif name == "list_rooms":
                return await self._handle_list_rooms(arguments)
            elif name == "get_room_members":
                return await self._handle_get_room_members(arguments)
            elif name == "supervisor_health":
                return await self._handle_supervisor_health(arguments)
            elif name == "get_agent_timeline":
                return await self._handle_get_agent_timeline(arguments)
            elif name == "get_agent_metrics":
                return await self._handle_get_agent_metrics(arguments)
            elif name == "search_decisions":
                return await self._handle_search_decisions(arguments)
            elif name == "send_room_message":
                return await self._handle_send_room_message(arguments)
            elif name == "get_room_messages":
                return await self._handle_get_room_messages(arguments)
            # Subroutine tools
            elif name == "run_agent_subroutine":
                return await self._handle_run_agent_subroutine(arguments)
            elif name == "check_agent_subroutine":
                return await self._handle_check_agent_subroutine(arguments)
            elif name == "list_agent_subroutines":
                return await self._handle_list_agent_subroutines(arguments)
            elif name == "get_agent_sleep_status":
                return await self._handle_get_agent_sleep_status(arguments)
            elif name == "get_self_version":
                return await self._handle_get_self_version(arguments)
            elif name == "request_upgrade":
                return await self._handle_request_upgrade(arguments)
            elif name == "queue_summary":
                return await self._handle_queue_summary(arguments)
            elif name == "queue_read":
                return await self._handle_queue_read(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_list_agents(self, arguments: dict) -> list[TextContent]:
        """Handle list_agents tool call.

        Returns:
            List containing single TextContent with JSON array of agent data
        """
        agents = self.state_manager.list_agents()

        # Convert AgentState objects to dicts
        agent_dicts = [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "template": a.template,
                "status": a.status,
                "container_id": a.container_id,
                "created_at": a.created_at.isoformat()
            }
            for a in agents
        ]

        response = {
            "agents": agent_dicts,
            "count": len(agent_dicts)
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    async def _handle_get_agent_details(self, arguments: dict) -> list[TextContent]:
        """Handle get_agent_details tool call.

        Args:
            arguments: Dict with agent_id key

        Returns:
            List containing single TextContent with detailed agent data
        """
        agent_id = arguments.get("agent_id")

        # Get agent from state
        agent = self.state_manager.get_agent(agent_id)

        if not agent:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Agent '{agent_id}' not found"})
            )]

        # Convert AgentState to dict
        agent_dict = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "template": agent.template,
            "status": agent.status,
            "container_id": agent.container_id,
            "config": agent.config,
            "public_key": agent.public_key,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat()
        }

        # Get container stats if running
        container_stats = None
        if agent.status == "running" and agent.container_id:
            try:
                container_stats = self.docker_manager.get_container_stats(agent.container_id)
            except Exception as e:
                container_stats = {"error": str(e)}

        # Get Matrix account if exists
        matrix_account = None
        try:
            matrix_account = self.matrix_manager.get_agent_account(agent_id)
        except Exception:
            pass  # Agent may not have Matrix account yet

        response = {
            **agent_dict,
            "container_stats": container_stats,
            "matrix_account": matrix_account
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    async def _handle_list_rooms(self, arguments: dict) -> list[TextContent]:
        """Handle list_rooms tool call.

        Args:
            arguments: Dict with optional include_archived boolean

        Returns:
            List containing single TextContent with JSON array of room data
        """
        include_archived = arguments.get("include_archived", False)

        rooms = self.matrix_manager.list_rooms(include_archived=include_archived)

        response = {
            "rooms": rooms,
            "count": len(rooms)
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    async def _handle_get_room_members(self, arguments: dict) -> list[TextContent]:
        """Handle get_room_members tool call.

        Args:
            arguments: Dict with room_identifier (room_id or alias)

        Returns:
            List containing single TextContent with member data
        """
        room_identifier = arguments.get("room_identifier")

        try:
            members = self.matrix_manager.get_room_members(room_identifier)

            response = {
                "room_identifier": room_identifier,
                "members": members,
                "count": len(members)
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    async def _handle_supervisor_health(self, arguments: dict) -> list[TextContent]:
        """Handle supervisor_health tool call.

        Returns:
            System health overview including agent counts, Matrix status, rooms
        """
        # Agent statistics
        agents = self.state_manager.list_agents()
        agent_stats = {
            "total": len(agents),
            "running": sum(1 for a in agents if a.status == "running"),
            "stopped": sum(1 for a in agents if a.status == "stopped"),
            "failed": sum(1 for a in agents if a.status == "failed")
        }

        # Matrix status
        synapse_running = self.matrix_manager.is_synapse_running()

        # Room statistics
        rooms = self.matrix_manager.list_rooms(include_archived=False)
        archived_rooms = self.matrix_manager.list_rooms(include_archived=True)
        room_stats = {
            "active": len(rooms),
            "archived": len(archived_rooms) - len(rooms)
        }

        # Docker network status
        network_status = self.docker_manager.get_network_status()

        response = {
            "status": "healthy" if synapse_running and network_status["exists"] else "degraded",
            "agents": agent_stats,
            "matrix": {
                "synapse_running": synapse_running
            },
            "rooms": room_stats,
            "docker": {
                "network_status": network_status
            }
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    async def _handle_get_agent_timeline(self, arguments: dict) -> list[TextContent]:
        """Handle get_agent_timeline tool call.

        Args:
            arguments: Dict with agent_id and optional filters

        Returns:
            List containing single TextContent with timeline data
        """
        if not self.observability_manager:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Observability not enabled"})
            )]

        agent_id = arguments.get("agent_id")
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        event_types = arguments.get("event_types")

        try:
            timeline = self.observability_manager.get_timeline(
                agent_id=agent_id,
                start_time=start_time,
                end_time=end_time,
                event_types=event_types
            )

            response = {
                "agent_id": agent_id,
                "timeline": timeline,
                "count": len(timeline)
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    async def _handle_get_agent_metrics(self, arguments: dict) -> list[TextContent]:
        """Handle get_agent_metrics tool call.

        Args:
            arguments: Dict with agent_id and optional filters

        Returns:
            List containing single TextContent with aggregated metrics
        """
        agent_id = arguments.get("agent_id")
        metric_types = arguments.get("metric_types")
        time_range_hours = arguments.get("time_range_hours", 24)

        try:
            client = OTelClient()
            metrics = client.get_agent_metrics(
                agent_id=agent_id,
                time_range_hours=time_range_hours,
                metric_types=metric_types
            )

            response = {
                "agent_id": agent_id,
                "time_range_hours": time_range_hours,
                "metrics": metrics
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    async def _handle_search_decisions(self, arguments: dict) -> list[TextContent]:
        """Handle search_decisions tool call.

        Args:
            arguments: Dict with optional agent_id, decision_type, and text_query

        Returns:
            List containing single TextContent with matching decision events
        """
        agent_id = arguments.get("agent_id")
        decision_type = arguments.get("decision_type")
        text_query = arguments.get("text_query")

        if not self.observability_manager:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Observability manager not available"})
            )]

        try:
            decisions = self.observability_manager.search_decision_events(
                agent_id=agent_id,
                decision_type=decision_type,
                text_query=text_query
            )

            response = {
                "count": len(decisions),
                "decisions": decisions
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    async def _handle_send_room_message(self, arguments: dict) -> list[TextContent]:
        """Handle send_room_message tool call.

        Args:
            arguments: Dict with room_id, message, and optional as_user

        Returns:
            List containing single TextContent with send confirmation
        """
        room_id = arguments.get("room_id")
        message = arguments.get("message")
        as_user = arguments.get("as_user", "supervisor")

        if not room_id or not message:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "room_id and message are required"})
            )]

        try:
            result = await self.matrix_manager.send_room_message(
                room_id=room_id,
                message=message,
                as_user=as_user
            )

            response = {
                "status": "sent",
                **result
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to send message: {str(e)}"})
            )]

    async def _handle_get_room_messages(self, arguments: dict) -> list[TextContent]:
        """Handle get_room_messages tool call.

        Args:
            arguments: Dict with room_id and optional limit, as_user

        Returns:
            List containing single TextContent with message history
        """
        room_id = arguments.get("room_id")
        limit = arguments.get("limit", 50)
        as_user = arguments.get("as_user", "supervisor")

        if not room_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "room_id is required"})
            )]

        try:
            messages = await self.matrix_manager.get_room_messages(
                room_id=room_id,
                limit=limit,
                as_user=as_user
            )

            response = {
                "room_id": room_id,
                "count": len(messages),
                "messages": messages
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to get messages: {str(e)}"})
            )]

    # ============ Subroutine Tools ============

    def _get_agent_container_url(self, agent_id: str) -> tuple[str, str]:
        """Get agent and validate it's running.

        Args:
            agent_id: Agent identifier

        Returns:
            Tuple of (container_url, error_json) where error_json is None if valid

        Raises:
            ValueError: If agent not found or not running
        """
        agent = self.state_manager.get_agent(agent_id)

        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")

        if agent.status != "running":
            raise ValueError(f"Agent '{agent_id}' is not running (status: {agent.status})")

        container_url = agent.config.get("container_url")
        if not container_url:
            raise ValueError(f"Agent '{agent_id}' has no container URL")

        return container_url

    async def _handle_run_agent_subroutine(self, arguments: dict) -> list[TextContent]:
        """Handle run_agent_subroutine tool call.

        Proxies the request to the agent's harness HTTP endpoint.

        Args:
            arguments: Dict with agent_id, subroutine_name, and optional params

        Returns:
            List containing single TextContent with task_id or error
        """
        agent_id = arguments.get("agent_id")
        subroutine_name = arguments.get("subroutine_name")
        params = arguments.get("params", {})

        try:
            container_url = self._get_agent_container_url(agent_id)

            response = requests.post(
                f"{container_url}/subroutines/run",
                json={"name": subroutine_name, "params": params},
                timeout=10
            )

            if response.status_code == 404:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Subroutine '{subroutine_name}' not found"})
                )]
            elif response.status_code == 400:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Subroutines are disabled for this agent"})
                )]

            response.raise_for_status()
            return [TextContent(
                type="text",
                text=json.dumps(response.json(), indent=2)
            )]

        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except requests.RequestException as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to contact agent: {str(e)}"})
            )]

    async def _handle_check_agent_subroutine(self, arguments: dict) -> list[TextContent]:
        """Handle check_agent_subroutine tool call.

        Proxies the request to the agent's harness HTTP endpoint.

        Args:
            arguments: Dict with agent_id and task_id

        Returns:
            List containing single TextContent with status or error
        """
        agent_id = arguments.get("agent_id")
        task_id = arguments.get("task_id")

        try:
            container_url = self._get_agent_container_url(agent_id)

            response = requests.get(
                f"{container_url}/subroutines/check/{task_id}",
                timeout=10
            )

            if response.status_code == 404:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Task '{task_id}' not found"})
                )]

            response.raise_for_status()
            return [TextContent(
                type="text",
                text=json.dumps(response.json(), indent=2)
            )]

        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except requests.RequestException as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to contact agent: {str(e)}"})
            )]

    async def _handle_list_agent_subroutines(self, arguments: dict) -> list[TextContent]:
        """Handle list_agent_subroutines tool call.

        Proxies the request to the agent's harness HTTP endpoint.

        Args:
            arguments: Dict with agent_id

        Returns:
            List containing single TextContent with available subroutines
        """
        agent_id = arguments.get("agent_id")

        try:
            container_url = self._get_agent_container_url(agent_id)

            response = requests.get(
                f"{container_url}/subroutines/list",
                timeout=10
            )

            response.raise_for_status()
            return [TextContent(
                type="text",
                text=json.dumps(response.json(), indent=2)
            )]

        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except requests.RequestException as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to contact agent: {str(e)}"})
            )]

    async def _handle_get_agent_sleep_status(self, arguments: dict) -> list[TextContent]:
        """Handle get_agent_sleep_status tool call.

        Proxies the request to the agent's harness HTTP endpoint.

        Args:
            arguments: Dict with agent_id

        Returns:
            List containing single TextContent with sleep status
        """
        agent_id = arguments.get("agent_id")

        try:
            container_url = self._get_agent_container_url(agent_id)

            response = requests.get(
                f"{container_url}/sleep/status",
                timeout=10
            )

            response.raise_for_status()
            return [TextContent(
                type="text",
                text=json.dumps(response.json(), indent=2)
            )]

        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except requests.RequestException as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to contact agent: {str(e)}"})
            )]

    async def _handle_get_self_version(self, arguments: dict) -> list[TextContent]:
        """Get agent version and upgrade history.

        Args:
            arguments: Dict with agent_id

        Returns:
            List containing single TextContent with version info

        Raises:
            ValueError: If agent_id is missing or agent not found
        """
        agent_id = arguments.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required")

        agent = self.state_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        history = self.state_manager.get_agent_upgrade_history(agent_id)

        # Extract version from image tag (e.g., "harness:agent-001-v2-passed" -> "v2")
        image_tag = agent.config.get("image_tag", "harness:base")
        version = "base"
        if "-v" in image_tag:
            parts = image_tag.split("-v")
            if len(parts) > 1:
                version_part = parts[-1].split("-")[0]  # Get number before -passed/-failed
                version = f"v{version_part}"

        # Get commit SHA from latest passed upgrade
        latest = self.state_manager.get_latest_passed_upgrade(agent_id)
        commit_sha = latest["commit_sha"] if latest else None

        response = {
            "current_version": version,
            "commit_sha": commit_sha,
            "image_tag": image_tag,
            "branch": f"agent/{agent_id}",
            "upgrade_history": [
                {
                    "version": f"v{i+1}",
                    "status": u["status"],
                    "commit": u["commit_sha"],
                    "reason": u.get("failure_reason")
                }
                for i, u in enumerate(reversed(history))
            ]
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    async def _handle_request_upgrade(self, arguments: dict) -> dict:
        """Handle agent upgrade request.

        Args:
            arguments: Dict with agent_id and commit_sha

        Returns:
            Dict with upgrade result

        Raises:
            ValueError: If agent_id or commit_sha is missing, or lifecycle_manager not available
        """
        agent_id = arguments.get("agent_id")
        commit_sha = arguments.get("commit_sha")

        if not agent_id or not commit_sha:
            raise ValueError("agent_id and commit_sha are required")

        if not self.lifecycle_manager:
            raise ValueError("Lifecycle manager not available")

        # Initiate upgrade (this will terminate the calling agent)
        result = await self.lifecycle_manager.upgrade_agent_from_branch(
            identifier=agent_id,
            commit_sha=commit_sha
        )

        return result

    def _verify_agent_signature(self, agent_id: str, signature: str) -> bool:
        """Verify agent's signature for queue access.

        Args:
            agent_id: Agent identifier
            signature: Signature in format "timestamp:base64_signature"

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Get agent's public key
            agent = self.state_manager.get_agent(agent_id)
            if not agent or not agent.public_key:
                return False

            # Parse signature
            parts = signature.split(":", 1)
            if len(parts) != 2:
                return False

            timestamp_str, sig_b64 = parts

            # Check timestamp is recent (within 5 minutes)
            timestamp = int(timestamp_str)
            now = int(time.time())
            if abs(now - timestamp) > 300:
                return False

            # Verify signature
            message = f"{agent_id}:{timestamp_str}".encode()
            sig_bytes = base64.b64decode(sig_b64)

            # Parse public key
            if not agent.public_key.startswith("ed25519:"):
                return False

            pub_b64 = agent.public_key.split(":")[1]
            pub_bytes = base64.b64decode(pub_b64)
            public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)

            public_key.verify(sig_bytes, message)
            return True

        except (ValueError, InvalidSignature, IndexError):
            return False

    async def _handle_queue_summary(self, arguments: dict) -> list[TextContent]:
        """Handle queue_summary tool call.

        Args:
            arguments: Dict with agent_id

        Returns:
            List containing single TextContent with queue summary
        """
        agent_id = arguments.get("agent_id")

        if not agent_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "agent_id is required"})
            )]

        agent = self.state_manager.get_agent(agent_id)
        if not agent:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Agent '{agent_id}' not found"})
            )]

        summary = self.state_manager.get_queue_summary(agent_id)

        return [TextContent(
            type="text",
            text=json.dumps(summary, indent=2)
        )]

    async def _handle_queue_read(self, arguments: dict) -> list[TextContent]:
        """Handle queue_read tool call with signature verification.

        Args:
            arguments: Dict with agent_id, optional signature, and optional filters

        Returns:
            List containing single TextContent with popped items
        """
        agent_id = arguments.get("agent_id")
        signature = arguments.get("signature")
        limit = arguments.get("limit")
        oldest_first = arguments.get("oldest_first", True)
        source = arguments.get("source")

        if not agent_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "agent_id is required"})
            )]

        agent = self.state_manager.get_agent(agent_id)
        if not agent:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Agent '{agent_id}' not found"})
            )]

        # Require signature if agent has public key
        if agent.public_key:
            if not signature:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Signature required for queue access"})
                )]

            if not self._verify_agent_signature(agent_id, signature):
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Invalid signature"})
                )]

        items = self.state_manager.pop_queue_items(
            agent_id=agent_id,
            limit=limit,
            oldest_first=oldest_first,
            source=source
        )

        response = {
            "agent_id": agent_id,
            "items": items,
            "count": len(items)
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    async def run(self):
        """Run the MCP server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

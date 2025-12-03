"""Supervisor JSON-RPC client"""

import requests
from typing import Any


class SupervisorError(Exception):
    """Error communicating with supervisor"""
    pass


class SupervisorClient:
    """Client for supervisor JSON-RPC API"""

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url
        self.rpc_url = f"{base_url}/rpc"
        self._request_id = 0

    def call(self, method: str, params: dict) -> Any:
        """Make JSON-RPC call to supervisor"""
        self._request_id += 1

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id
        }

        try:
            response = requests.post(self.rpc_url, json=payload, timeout=120)
            response.raise_for_status()
        except Exception as e:
            raise SupervisorError(f"Failed to connect to supervisor: {e}")

        data = response.json()

        if "error" in data:
            error = data["error"]
            raise SupervisorError(f"{error.get('message', 'Unknown error')}")

        return data.get("result")

    def list_agents(self) -> list[dict]:
        """List all agents"""
        return self.call("list_agents", {})

    def spawn_agent(self, template: str, agent_id: str = None) -> dict:
        """Spawn an agent from template"""
        params = {"template": template}
        if agent_id:
            params["agent_id"] = agent_id
        return self.call("spawn_agent", params)

    def send_message(self, agent_id: str, message: str) -> dict:
        """Send message to agent"""
        return self.call("send_message", {"agent_id": agent_id, "message": message})

    def stop_agent(self, agent_id: str) -> dict:
        """Stop an agent"""
        return self.call("stop_agent", {"agent_id": agent_id})

    def start_agent(self, agent_id: str) -> dict:
        """Start a stopped agent"""
        return self.call("start_agent", {"agent_id": agent_id})

    def remove_agent(self, agent_id: str) -> dict:
        """Remove an agent completely"""
        return self.call("remove_agent", {"agent_id": agent_id})

    def upgrade_agent(
        self,
        agent_id: str,
        rebuild: bool = False,
        refresh_config: bool = False
    ) -> dict:
        """Upgrade an agent's container"""
        return self.call("upgrade_agent", {
            "agent_id": agent_id,
            "rebuild": rebuild,
            "refresh_config": refresh_config
        })

    def health_check(self) -> bool:
        """Check if supervisor is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    # Room operations (HTTP REST endpoints, not JSON-RPC)
    def create_room(self, name: str, topic: str = None) -> dict:
        """Create a Matrix room"""
        payload = {"name": name}
        if topic:
            payload["topic"] = topic

        try:
            response = requests.post(f"{self.base_url}/rooms/create", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise SupervisorError(f"Failed to create room: {e}")

    def invite_agent_to_room(self, room_id: str, agent_id: str) -> dict:
        """Invite agent to room by agent_id"""
        payload = {"room_id": room_id, "agent_id": agent_id}

        try:
            response = requests.post(f"{self.base_url}/rooms/invite", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise SupervisorError(f"Failed to invite agent to room: {e}")

    def get_room_members(self, room_id: str) -> list[dict]:
        """Get room members"""
        payload = {"room_id": room_id}

        try:
            response = requests.post(f"{self.base_url}/rooms/members", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("members", [])
        except Exception as e:
            raise SupervisorError(f"Failed to get room members: {e}")

    def list_rooms(self, include_archived: bool = False) -> list[dict]:
        """List rooms"""
        try:
            response = requests.get(
                f"{self.base_url}/rooms",
                params={"include_archived": include_archived},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("rooms", [])
        except Exception as e:
            raise SupervisorError(f"Failed to list rooms: {e}")

    def delete_room(self, room_id: str) -> dict:
        """Delete (archive) a room"""
        payload = {"room_id": room_id}

        try:
            response = requests.post(f"{self.base_url}/rooms/delete", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise SupervisorError(f"Failed to delete room: {e}")

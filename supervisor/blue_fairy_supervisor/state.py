"""SQLite-based state management for agents"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from blue_fairy_common.types import AgentStatus, InvalidTransition, validate_transition


@dataclass
class AgentState:
    """Agent state record"""
    agent_id: str
    name: Optional[str]
    template: str
    container_id: str
    status: str
    config: dict
    observability_level: int = 0
    self_reflection_enabled: bool = False
    observability_config: str = '{}'
    error_message: Optional[str] = None
    provisioned_resources: str = '{}'
    public_key: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None


class StateManager:
    """Manage agent state in SQLite database"""

    def __init__(self, db_path: Path | str):
        """Initialize state manager with database path"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_db()

    def _initialize_db(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                template TEXT NOT NULL,
                container_id TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                observability_level INTEGER DEFAULT 0,
                self_reflection_enabled BOOLEAN DEFAULT FALSE,
                observability_config TEXT DEFAULT '{}',
                error_message TEXT,
                provisioned_resources TEXT DEFAULT '{}',
                public_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate existing tables by adding missing columns
        cursor.execute("PRAGMA table_info(agents)")
        columns = {row[1] for row in cursor.fetchall()}

        if "error_message" not in columns:
            cursor.execute("ALTER TABLE agents ADD COLUMN error_message TEXT")

        if "provisioned_resources" not in columns:
            cursor.execute("ALTER TABLE agents ADD COLUMN provisioned_resources TEXT DEFAULT '{}'")

        if "public_key" not in columns:
            cursor.execute("ALTER TABLE agents ADD COLUMN public_key TEXT")


        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message TEXT NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matrix_supervisor (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                user_id TEXT NOT NULL,
                access_token TEXT NOT NULL,
                registration_secret TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Rooms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                room_id TEXT PRIMARY KEY,
                room_name TEXT NOT NULL,
                room_alias TEXT,
                topic TEXT,
                created_by TEXT,
                created_at INTEGER NOT NULL,
                is_archived INTEGER DEFAULT 0
            )
        """)

        # Room membership table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS room_members (
                room_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                member_type TEXT NOT NULL,
                agent_id TEXT,
                joined_at INTEGER NOT NULL,
                left_at INTEGER,
                PRIMARY KEY (room_id, user_id),
                FOREIGN KEY (room_id) REFERENCES rooms(room_id) ON DELETE CASCADE
            )
        """)

        # Agent Matrix accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matrix_accounts (
                agent_id TEXT PRIMARY KEY,
                matrix_user_id TEXT NOT NULL UNIQUE,
                access_token TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
            )
        """)

        # Agent upgrades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_upgrades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                image_tag TEXT NOT NULL,
                previous_image_tag TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                continuity_secret TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                failure_reason TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        """)

        # Agent deploy keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_deploy_keys (
                agent_id TEXT PRIMARY KEY,
                github_key_id INTEGER NOT NULL,
                public_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
            )
        """)

        # Agent event queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_queue (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_queue_agent
            ON agent_queue(agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_queue_timestamp
            ON agent_queue(agent_id, timestamp)
        """)

        self.conn.commit()

    def create_agent(
        self,
        agent_id: str,
        template: str,
        container_id: str,
        config: dict,
        observability_level: int = 0,
        self_reflection_enabled: bool = False,
        observability_config: str = '{}',
        status: str = "running"
    ) -> AgentState:
        """Create new agent record

        Args:
            agent_id: Unique agent identifier
            template: Template name used to spawn agent
            container_id: Docker container ID
            config: Agent configuration dictionary
            observability_level: 0=private, 1=metrics, 2=full (default: 0)
            self_reflection_enabled: Allow agent self-introspection (default: False)
            observability_config: JSON string of observability event preferences
            status: Initial agent status (default: "running" for backward compatibility)
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO agents (
                agent_id, template, container_id, status, config_json,
                observability_level, self_reflection_enabled, observability_config
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (agent_id, template, container_id, status, json.dumps(config),
              observability_level, self_reflection_enabled, observability_config))

        self.conn.commit()

        return self.get_agent(agent_id)

    def get_agent(self, identifier: str) -> Optional[AgentState]:
        """Get agent by agent_id or name"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM agents WHERE agent_id = ? OR name = ?
        """, (identifier, identifier))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_agent(row)

    def list_agents(self, status: Optional[str] = None) -> list[AgentState]:
        """List all agents, optionally filtered by status"""
        cursor = self.conn.cursor()

        if status:
            cursor.execute("SELECT * FROM agents WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT * FROM agents")

        return [self._row_to_agent(row) for row in cursor.fetchall()]

    def get_existing_names(self) -> list[str]:
        """Get list of all agent names (excluding None)"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM agents WHERE name IS NOT NULL ORDER BY name")
        return [row["name"] for row in cursor.fetchall()]

    def update_agent_name(self, agent_id: str, name: str):
        """Update agent's chosen name"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE agents SET name = ?, updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ?
        """, (name, agent_id))
        self.conn.commit()

    def update_agent_public_key(self, agent_id: str, public_key: str):
        """Update agent's public key (cryptographic identity)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE agents SET public_key = ?, updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ?
        """, (public_key, agent_id))
        self.conn.commit()

    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE agents SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ?
        """, (status, agent_id))
        self.conn.commit()

    def update_agent_container(self, agent_id: str, container_id: str):
        """Update agent's container ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE agents SET container_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ?
        """, (container_id, agent_id))
        self.conn.commit()

    def update_agent_config(self, agent_id: str, config: dict):
        """Update agent configuration"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE agents SET config_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ?
        """, (json.dumps(config), agent_id))
        self.conn.commit()

    def update_agent_observability(
        self,
        agent_id: str,
        observability_level: int,
        self_reflection_enabled: bool,
        observability_config: str
    ) -> None:
        """Update agent's observability settings

        Args:
            agent_id: Agent identifier
            observability_level: 0=private, 1=metrics, 2=full
            self_reflection_enabled: Allow agent self-introspection
            observability_config: JSON string of event preferences
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE agents
            SET observability_level = ?,
                self_reflection_enabled = ?,
                observability_config = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ?
        """, (observability_level, self_reflection_enabled, observability_config, agent_id))
        self.conn.commit()

    def delete_agent(self, agent_id: str):
        """Delete agent record"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
        self.conn.commit()

    def set_error(self, agent_id: str, message: str) -> None:
        """Set error message for failed agent"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE agents SET error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE agent_id = ?",
            (message, agent_id)
        )
        self.conn.commit()

    def set_provisioned_resources(self, agent_id: str, resources: dict) -> None:
        """Track provisioned resources for rollback"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE agents SET provisioned_resources = ?, updated_at = CURRENT_TIMESTAMP WHERE agent_id = ?",
            (json.dumps(resources), agent_id)
        )
        self.conn.commit()

    def get_provisioned_resources(self, agent_id: str) -> dict:
        """Get provisioned resources for cleanup"""
        agent = self.get_agent(agent_id)
        if not agent:
            return {}
        try:
            return json.loads(agent.provisioned_resources or "{}")
        except json.JSONDecodeError:
            return {}

    def transition(self, agent_id: str, new_status: AgentStatus) -> None:
        """Transition agent to new status with validation.

        Raises:
            InvalidTransition: If transition is not valid from current state
            ValueError: If agent not found
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        current = AgentStatus(agent.status)
        validate_transition(current, new_status)

        self.update_agent_status(agent_id, new_status.value)

    def _row_to_agent(self, row: sqlite3.Row) -> AgentState:
        """Convert database row to AgentState"""
        return AgentState(
            agent_id=row["agent_id"],
            name=row["name"],
            template=row["template"],
            container_id=row["container_id"],
            status=row["status"],
            config=json.loads(row["config_json"]),
            observability_level=row["observability_level"],
            self_reflection_enabled=bool(row["self_reflection_enabled"]),
            observability_config=row["observability_config"],
            error_message=row["error_message"],
            provisioned_resources=row["provisioned_resources"] or "{}",
            public_key=row["public_key"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def store_supervisor_matrix_credentials(
        self,
        user_id: str,
        access_token: str,
        registration_secret: str
    ):
        """Store supervisor Matrix credentials (upsert)"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO matrix_supervisor (id, user_id, access_token, registration_secret)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id = excluded.user_id,
                access_token = excluded.access_token,
                registration_secret = excluded.registration_secret,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, access_token, registration_secret))

        self.conn.commit()

    def get_supervisor_matrix_credentials(self) -> Optional[dict]:
        """Get supervisor Matrix credentials"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT user_id, access_token, registration_secret
            FROM matrix_supervisor
            WHERE id = 1
        """)

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "user_id": row["user_id"],
            "access_token": row["access_token"],
            "registration_secret": row["registration_secret"]
        }

    def add_room(
        self,
        room_id: str,
        room_name: str,
        room_alias: str,
        topic: Optional[str],
        created_by: str
    ):
        """Add a new room to the database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO rooms (room_id, room_name, room_alias, topic, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (room_id, room_name, room_alias, topic, created_by, int(time.time()))
        )
        self.conn.commit()

    def get_room(self, room_id: str) -> Optional[dict]:
        """Get room metadata by room_id"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM rooms WHERE room_id = ?",
            (room_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "room_id": row["room_id"],
                "room_name": row["room_name"],
                "room_alias": row["room_alias"],
                "topic": row["topic"],
                "created_by": row["created_by"],
                "created_at": row["created_at"],
                "is_archived": row["is_archived"]
            }
        return None

    def list_rooms(self, include_archived: bool = False) -> list[dict]:
        """List all rooms"""
        cursor = self.conn.cursor()
        if include_archived:
            cursor.execute("SELECT * FROM rooms ORDER BY created_at DESC")
        else:
            cursor.execute(
                "SELECT * FROM rooms WHERE is_archived = 0 ORDER BY created_at DESC"
            )

        rooms = []
        for row in cursor.fetchall():
            rooms.append({
                "room_id": row["room_id"],
                "room_name": row["room_name"],
                "room_alias": row["room_alias"],
                "topic": row["topic"],
                "created_by": row["created_by"],
                "created_at": row["created_at"],
                "is_archived": row["is_archived"]
            })
        return rooms

    def archive_room(self, room_id: str):
        """Mark a room as archived"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE rooms SET is_archived = 1 WHERE room_id = ?",
            (room_id,)
        )
        self.conn.commit()

    def add_room_member(
        self,
        room_id: str,
        user_id: str,
        member_type: str,
        agent_id: Optional[str] = None
    ):
        """Add a member to a room"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO room_members (room_id, user_id, member_type, agent_id, joined_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(room_id, user_id) DO UPDATE SET
                left_at = NULL,
                joined_at = ?
            """,
            (room_id, user_id, member_type, agent_id, int(time.time()), int(time.time()))
        )
        self.conn.commit()

    def remove_room_member(self, room_id: str, user_id: str):
        """Mark a member as having left the room"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE room_members SET left_at = ? WHERE room_id = ? AND user_id = ?",
            (int(time.time()), room_id, user_id)
        )
        self.conn.commit()

    def get_room_members(
        self,
        room_id: str,
        include_left: bool = False
    ) -> list[dict]:
        """Get all members of a room"""
        cursor = self.conn.cursor()
        if include_left:
            cursor.execute(
                "SELECT * FROM room_members WHERE room_id = ?",
                (room_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM room_members WHERE room_id = ? AND left_at IS NULL",
                (room_id,)
            )

        members = []
        for row in cursor.fetchall():
            members.append({
                "room_id": row["room_id"],
                "user_id": row["user_id"],
                "member_type": row["member_type"],
                "agent_id": row["agent_id"],
                "joined_at": row["joined_at"],
                "left_at": row["left_at"]
            })
        return members

    def store_agent_matrix_credentials(
        self,
        agent_id: str,
        matrix_user_id: str,
        access_token: str
    ):
        """Store Matrix credentials for an agent"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO matrix_accounts (agent_id, matrix_user_id, access_token, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                matrix_user_id = excluded.matrix_user_id,
                access_token = excluded.access_token
            """,
            (agent_id, matrix_user_id, access_token, int(time.time()))
        )
        self.conn.commit()

    def get_agent_matrix_credentials(self, agent_id: str) -> dict | None:
        """Get Matrix credentials for an agent"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id, matrix_user_id, access_token, created_at FROM matrix_accounts WHERE agent_id = ?",
            (agent_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "agent_id": row["agent_id"],
                "matrix_user_id": row["matrix_user_id"],
                "access_token": row["access_token"],
                "created_at": row["created_at"]
            }
        return None

    def delete_agent_matrix_credentials(self, agent_id: str):
        """Delete Matrix credentials for an agent"""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM matrix_accounts WHERE agent_id = ?",
            (agent_id,)
        )
        self.conn.commit()

    def create_upgrade_record(
        self,
        agent_id: str,
        commit_sha: str,
        image_tag: str,
        previous_image_tag: str,
        continuity_secret: str
    ) -> int:
        """Create a new upgrade record for an agent

        Args:
            agent_id: Agent identifier
            commit_sha: Git commit SHA for this upgrade
            image_tag: Docker image tag for new version
            previous_image_tag: Docker image tag being upgraded from
            continuity_secret: Secret token for maintaining agent identity

        Returns:
            ID of the created upgrade record
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_upgrades (
                agent_id, commit_sha, image_tag, previous_image_tag,
                continuity_secret, status
            )
            VALUES (?, ?, ?, ?, ?, 'pending')
            """,
            (agent_id, commit_sha, image_tag, previous_image_tag, continuity_secret)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_upgrade_record(self, upgrade_id: int) -> dict | None:
        """Get upgrade record by ID

        Args:
            upgrade_id: Upgrade record ID

        Returns:
            Upgrade record dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, agent_id, commit_sha, image_tag, previous_image_tag,
                   status, continuity_secret, created_at, completed_at, failure_reason
            FROM agent_upgrades
            WHERE id = ?
            """,
            (upgrade_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "agent_id": row["agent_id"],
                "commit_sha": row["commit_sha"],
                "image_tag": row["image_tag"],
                "previous_image_tag": row["previous_image_tag"],
                "status": row["status"],
                "continuity_secret": row["continuity_secret"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "failure_reason": row["failure_reason"]
            }
        return None

    def update_upgrade_status(
        self,
        upgrade_id: int,
        status: str,
        failure_reason: str | None = None
    ) -> None:
        """Update upgrade status

        Args:
            upgrade_id: Upgrade record ID
            status: New status (pending, in_progress, passed, failed)
            failure_reason: Optional reason if status is failed
        """
        cursor = self.conn.cursor()

        # Set completed_at timestamp for terminal states (passed, failed)
        if status in ("passed", "failed"):
            cursor.execute(
                """
                UPDATE agent_upgrades
                SET status = ?, completed_at = CURRENT_TIMESTAMP, failure_reason = ?
                WHERE id = ?
                """,
                (status, failure_reason, upgrade_id)
            )
        else:
            cursor.execute(
                """
                UPDATE agent_upgrades
                SET status = ?, failure_reason = ?
                WHERE id = ?
                """,
                (status, failure_reason, upgrade_id)
            )
        self.conn.commit()

    def get_agent_upgrade_history(self, agent_id: str) -> list[dict]:
        """Get upgrade history for an agent, most recent first

        Args:
            agent_id: Agent identifier

        Returns:
            List of upgrade records, ordered by created_at DESC, then id DESC
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, agent_id, commit_sha, image_tag, previous_image_tag,
                   status, continuity_secret, created_at, completed_at, failure_reason
            FROM agent_upgrades
            WHERE agent_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (agent_id,)
        )

        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row["id"],
                "agent_id": row["agent_id"],
                "commit_sha": row["commit_sha"],
                "image_tag": row["image_tag"],
                "previous_image_tag": row["previous_image_tag"],
                "status": row["status"],
                "continuity_secret": row["continuity_secret"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "failure_reason": row["failure_reason"]
            })
        return history

    def get_latest_passed_upgrade(self, agent_id: str) -> dict | None:
        """Get the most recent passed upgrade for an agent

        Args:
            agent_id: Agent identifier

        Returns:
            Latest passed upgrade record or None if no passed upgrades
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, agent_id, commit_sha, image_tag, previous_image_tag,
                   status, continuity_secret, created_at, completed_at, failure_reason
            FROM agent_upgrades
            WHERE agent_id = ? AND status = 'passed'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (agent_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "agent_id": row["agent_id"],
                "commit_sha": row["commit_sha"],
                "image_tag": row["image_tag"],
                "previous_image_tag": row["previous_image_tag"],
                "status": row["status"],
                "continuity_secret": row["continuity_secret"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "failure_reason": row["failure_reason"]
            }
        return None

    def create_deploy_key_record(
        self,
        agent_id: str,
        github_key_id: int,
        public_key: str
    ) -> None:
        """Store deploy key info for an agent

        Args:
            agent_id: Agent identifier
            github_key_id: GitHub's ID for the deploy key
            public_key: SSH public key content
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_deploy_keys (agent_id, github_key_id, public_key)
            VALUES (?, ?, ?)
            """,
            (agent_id, github_key_id, public_key)
        )
        self.conn.commit()

    def get_deploy_key(self, agent_id: str) -> dict | None:
        """Retrieve deploy key info for an agent

        Args:
            agent_id: Agent identifier

        Returns:
            Deploy key record dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT agent_id, github_key_id, public_key, created_at
            FROM agent_deploy_keys
            WHERE agent_id = ?
            """,
            (agent_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "agent_id": row["agent_id"],
                "github_key_id": row["github_key_id"],
                "public_key": row["public_key"],
                "created_at": row["created_at"]
            }
        return None

    def delete_deploy_key_record(self, agent_id: str) -> None:
        """Delete deploy key record for an agent

        Args:
            agent_id: Agent identifier
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM agent_deploy_keys WHERE agent_id = ?",
            (agent_id,)
        )
        self.conn.commit()

    def push_queue_item(
        self,
        agent_id: str,
        source: str,
        summary: str
    ) -> str:
        """Push an event to agent's queue.

        Args:
            agent_id: Agent identifier
            source: Event source (e.g., "chat.matrix", "system")
            summary: Human-readable summary

        Returns:
            Generated item ID
        """
        item_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_queue (id, agent_id, source, summary)
            VALUES (?, ?, ?, ?)
            """,
            (item_id, agent_id, source, summary)
        )
        self.conn.commit()
        return item_id

    def get_queue_summary(self, agent_id: str) -> dict:
        """Get summary of agent's queue.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with total count and counts by source
        """
        cursor = self.conn.cursor()

        # Get total
        cursor.execute(
            "SELECT COUNT(*) FROM agent_queue WHERE agent_id = ?",
            (agent_id,)
        )
        total = cursor.fetchone()[0]

        # Get counts by source
        cursor.execute(
            """
            SELECT source, COUNT(*) as count
            FROM agent_queue
            WHERE agent_id = ?
            GROUP BY source
            """,
            (agent_id,)
        )
        by_source = {row[0]: row[1] for row in cursor.fetchall()}

        return {"total": total, "by_source": by_source}

    def pop_queue_items(
        self,
        agent_id: str,
        limit: int | None = None,
        oldest_first: bool = True,
        source: str | None = None
    ) -> list[dict]:
        """Pop items from agent's queue (destructive read).

        Args:
            agent_id: Agent identifier
            limit: Max items to return (None = all)
            oldest_first: If True, return oldest items first
            source: Filter by source (None = all sources)

        Returns:
            List of queue items (removed from queue)
        """
        cursor = self.conn.cursor()

        # Build query
        query = "SELECT id, source, summary, timestamp FROM agent_queue WHERE agent_id = ?"
        params = [agent_id]

        if source:
            query += " AND source = ?"
            params.append(source)

        order = "ASC" if oldest_first else "DESC"
        query += f" ORDER BY timestamp {order}"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        items = []
        ids_to_delete = []
        for row in rows:
            items.append({
                "id": row[0],
                "source": row[1],
                "summary": row[2],
                "timestamp": row[3]
            })
            ids_to_delete.append(row[0])

        # Delete the items we're returning
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            cursor.execute(
                f"DELETE FROM agent_queue WHERE id IN ({placeholders})",
                ids_to_delete
            )
            self.conn.commit()

        return items

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed"""
        self.close()
        return False

    def close(self):
        """Close database connection"""
        self.conn.close()

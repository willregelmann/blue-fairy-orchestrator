"""Observability data collection and storage"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class ObservabilityManager:
    """Manages observability database for agent metrics and telemetry events"""

    def __init__(self, db_path: Path):
        """Initialize observability database

        Args:
            db_path: Path to observability.db file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._initialize_db()

    def _initialize_db(self):
        """Create tables and indexes for observability data"""
        cursor = self.conn.cursor()

        # Agent metrics table (Level 1 and Level 2)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                value REAL,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        """)

        # Telemetry events table (Level 2 only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        """)

        # Indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_agent_time
            ON agent_metrics(agent_id, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_type
            ON agent_metrics(metric_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_agent_time
            ON telemetry_events(agent_id, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON telemetry_events(event_type)
        """)

        self.conn.commit()

    def store_metric(
        self,
        agent_id: str,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a metric for an agent

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric (e.g., 'message_received', 'response_time')
            value: Numeric value of metric
            metadata: Optional additional context as dict
        """
        cursor = self.conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO agent_metrics (agent_id, metric_type, value, metadata)
            VALUES (?, ?, ?, ?)
        """, (agent_id, metric_type, value, metadata_json))

        self.conn.commit()

    def store_event(
        self,
        agent_id: str,
        event_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Store a telemetry event for an agent

        Args:
            agent_id: Agent identifier
            event_id: Unique event identifier (UUID from agent)
            event_type: Type of event ('decision', 'memory_retrieval', 'communication', 'reasoning')
            data: Event payload as dict
        """
        cursor = self.conn.cursor()
        data_json = json.dumps(data)

        cursor.execute("""
            INSERT INTO telemetry_events (agent_id, event_id, event_type, data)
            VALUES (?, ?, ?, ?)
        """, (agent_id, event_id, event_type, data_json))

        self.conn.commit()

    def get_timeline(
        self,
        agent_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get chronological timeline of events and metrics for an agent

        Args:
            agent_id: Agent identifier
            start_time: Start timestamp (ISO format), optional
            end_time: End timestamp (ISO format), optional
            event_types: Filter by event types (e.g., ['decision', 'memory_retrieval']), optional

        Returns:
            List of timeline items (events and metrics) ordered by timestamp
        """
        cursor = self.conn.cursor()
        timeline = []

        # Query events
        event_query = "SELECT event_id, timestamp, event_type, data FROM telemetry_events WHERE agent_id = ?"
        event_params = [agent_id]

        if start_time:
            event_query += " AND timestamp >= ?"
            event_params.append(start_time)

        if end_time:
            event_query += " AND timestamp <= ?"
            event_params.append(end_time)

        if event_types:
            placeholders = ','.join('?' * len(event_types))
            event_query += f" AND event_type IN ({placeholders})"
            event_params.extend(event_types)

        cursor.execute(event_query, event_params)
        for row in cursor.fetchall():
            timeline.append({
                "type": "event",
                "event_id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "data": json.loads(row[3])
            })

        # Query metrics (if no event_type filter specified)
        if not event_types:
            metric_query = "SELECT timestamp, metric_type, value, metadata FROM agent_metrics WHERE agent_id = ?"
            metric_params = [agent_id]

            if start_time:
                metric_query += " AND timestamp >= ?"
                metric_params.append(start_time)

            if end_time:
                metric_query += " AND timestamp <= ?"
                metric_params.append(end_time)

            cursor.execute(metric_query, metric_params)
            for row in cursor.fetchall():
                timeline.append({
                    "type": "metric",
                    "timestamp": row[0],
                    "metric_type": row[1],
                    "value": row[2],
                    "metadata": json.loads(row[3]) if row[3] else None
                })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def get_metrics_summary(
        self,
        agent_id: str,
        metric_types: Optional[List[str]] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics summary for an agent

        Args:
            agent_id: Agent identifier
            metric_types: Filter by metric types, optional (None = all types)
            time_range_hours: Time range in hours (default: 24)

        Returns:
            Dict mapping metric_type to aggregated stats (count, sum, avg, min, max)
        """
        cursor = self.conn.cursor()

        # Calculate start time
        from datetime import datetime, timedelta, timezone
        start_time = (datetime.now(timezone.utc) - timedelta(hours=time_range_hours)).isoformat()

        # Build query
        query = """
            SELECT metric_type,
                   COUNT(*) as count,
                   SUM(value) as sum,
                   AVG(value) as avg,
                   MIN(value) as min,
                   MAX(value) as max
            FROM agent_metrics
            WHERE agent_id = ?
            AND timestamp >= ?
        """
        params = [agent_id, start_time]

        if metric_types:
            placeholders = ','.join('?' * len(metric_types))
            query += f" AND metric_type IN ({placeholders})"
            params.extend(metric_types)

        query += " GROUP BY metric_type"

        cursor.execute(query, params)

        summary = {}
        for row in cursor.fetchall():
            metric_type = row[0]
            summary[metric_type] = {
                "count": row[1],
                "sum": row[2],
                "avg": row[3],
                "min": row[4],
                "max": row[5]
            }

        return summary

    def search_decision_events(
        self,
        agent_id: Optional[str] = None,
        decision_type: Optional[str] = None,  # 'respond' or 'silent'
        text_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search agent decision events with filters

        Args:
            agent_id: Filter by agent, optional (None = all agents)
            decision_type: Filter by decision type ('respond' or 'silent'), optional
            text_query: Search in reasoning/message text, optional

        Returns:
            List of decision events matching filters
        """
        cursor = self.conn.cursor()

        query = "SELECT agent_id, event_id, timestamp, data FROM telemetry_events WHERE event_type = 'decision'"
        params = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            data = json.loads(row[3])

            # Filter by decision type
            if decision_type and data.get("decision") != decision_type:
                continue

            # Filter by text query (search in reasoning and message)
            if text_query:
                searchable_text = f"{data.get('reasoning', '')} {data.get('message', '')}".lower()
                if text_query.lower() not in searchable_text:
                    continue

            results.append({
                "agent_id": row[0],
                "event_id": row[1],
                "timestamp": row[2],
                "data": data
            })

        return results

    def close(self):
        """Close database connection"""
        self.conn.close()


class MetricsCollector:
    """Collects passive metrics for Level 1 agents"""

    def __init__(self, observability_manager: ObservabilityManager):
        """Initialize metrics collector

        Args:
            observability_manager: ObservabilityManager instance for storing metrics
        """
        self.obs = observability_manager

    def record_message_received(
        self,
        agent_id: str,
        room_id: str,
        sender: str
    ) -> None:
        """Record that agent received a message

        Args:
            agent_id: Agent that received message
            room_id: Room where message was received
            sender: User ID of message sender
        """
        self.obs.store_metric(
            agent_id=agent_id,
            metric_type="message_received",
            value=1.0,
            metadata={"room_id": room_id, "sender": sender}
        )

    def record_message_sent(
        self,
        agent_id: str,
        room_id: str,
        message_length: int
    ) -> None:
        """Record that agent sent a message

        Args:
            agent_id: Agent that sent message
            room_id: Room where message was sent
            message_length: Length of message in characters
        """
        self.obs.store_metric(
            agent_id=agent_id,
            metric_type="message_sent",
            value=1.0,
            metadata={"room_id": room_id, "message_length": message_length}
        )

    def record_response_time(
        self,
        agent_id: str,
        response_time_seconds: float
    ) -> None:
        """Record time taken for agent to respond

        Args:
            agent_id: Agent identifier
            response_time_seconds: Time in seconds from message received to response sent
        """
        self.obs.store_metric(
            agent_id=agent_id,
            metric_type="response_time",
            value=response_time_seconds
        )

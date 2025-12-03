"""Client for querying OpenTelemetry backends (Tempo, Prometheus)."""

import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class OTelClient:
    """Client for Tempo and Prometheus APIs."""

    def __init__(
        self,
        tempo_url: str = "http://localhost:3200",
        prometheus_url: str = "http://localhost:9090"
    ):
        self.tempo_url = tempo_url
        self.prometheus_url = prometheus_url

    def search_decisions(
        self,
        agent_id: str,
        decision_type: str = None,
        text_query: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for decision events in Tempo traces.

        Returns: [{timestamp, agent_id, decision_type, reasoning, trace_id}]
        """
        try:
            # Build TraceQL query
            query = f'''{{
                resource.service.name="blue-fairy-agent"
                && span.agent_id="{agent_id}"
                && name="decision.evaluate"
            }}'''

            if decision_type:
                query += f' | {{ span.decision_type="{decision_type}" }}'

            # Query Tempo
            response = requests.post(
                f"{self.tempo_url}/api/search",
                json={"query": query},
                timeout=5
            )
            response.raise_for_status()

            # Extract decision events from traces
            decisions = []
            for trace_result in response.json().get("traces", []):
                trace_id = trace_result.get("traceID")

                # Fetch full trace
                trace_response = requests.get(
                    f"{self.tempo_url}/api/traces/{trace_id}",
                    timeout=5
                )
                trace_response.raise_for_status()

                # Extract decision reasoning from span events
                trace_data = trace_response.json()
                for batch in trace_data.get("batches", []):
                    for span in batch.get("spans", []):
                        if span.get("name") == "decision.evaluate":
                            for event in span.get("events", []):
                                if event.get("name") == "decision_reasoning":
                                    reasoning = event.get("attributes", {}).get("reasoning")

                                    # Filter by text query if provided
                                    if text_query and text_query.lower() not in reasoning.lower():
                                        continue

                                    decisions.append({
                                        "timestamp": span.get("startTimeUnixNano"),
                                        "agent_id": agent_id,
                                        "decision_type": span.get("attributes", {}).get("decision.type"),
                                        "reasoning": reasoning,
                                        "trace_id": trace_id
                                    })

            return decisions

        except Exception as e:
            logger.warning(f"Failed to query Tempo for decisions: {e}")
            return []

    def get_agent_metrics(
        self,
        agent_id: str,
        time_range_hours: int = 24,
        metric_types: List[str] = None
    ) -> Dict[str, Any]:
        """Query Prometheus for agent metrics."""
        try:
            metrics = {}

            # Default metrics to query
            if not metric_types:
                metric_types = ["message_sent", "response_time", "decision_counts"]

            for metric_type in metric_types:
                if metric_type == "message_sent":
                    query = f'sum(increase(agent_messages_sent_total{{agent_id="{agent_id}"}}[{time_range_hours}h]))'
                elif metric_type == "response_time":
                    query = f'histogram_quantile(0.5, agent_response_latency_bucket{{agent_id="{agent_id}"}})'
                elif metric_type == "decision_counts":
                    query = f'sum by (decision_type) (agent_decisions_total{{agent_id="{agent_id}"}})'
                else:
                    continue

                response = requests.post(
                    f"{self.prometheus_url}/api/v1/query",
                    data={"query": query},
                    timeout=5
                )
                response.raise_for_status()

                result = response.json().get("data", {}).get("result", [])
                metrics[metric_type] = result

            return metrics

        except Exception as e:
            logger.warning(f"Failed to query Prometheus for metrics: {e}")
            return {}

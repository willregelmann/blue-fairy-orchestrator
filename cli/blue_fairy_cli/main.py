"""CLI command definitions"""

import os
import sys
import click
from rich.console import Console
from rich.table import Table

from .client import SupervisorClient, SupervisorError
from .daemon import DaemonManager


console = Console()

# API base URL for supervisor
API_BASE = os.getenv("BLUE_FAIRY_API", "http://localhost:8765")


@click.group()
def cli():
    """Blue Fairy - AI agent orchestration"""
    pass


@cli.command()
@click.argument('template')
@click.option('--id', 'agent_id', help='Specific agent ID (auto-generated if not provided)')
def spawn(template, agent_id):
    """Spawn an agent from a template"""
    daemon = DaemonManager()

    # Ensure supervisor is running
    if not daemon.ensure_running():
        console.print("[red]Failed to start supervisor daemon[/red]")
        sys.exit(1)

    client = SupervisorClient()

    try:
        result = client.spawn_agent(template, agent_id)
        console.print(f"[green]Agent '{result['name']}' ({result['agent_id']}) is running[/green]")
    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def ps(verbose):
    """List running agents"""
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[yellow]Supervisor is not running[/yellow]")
        return

    client = SupervisorClient()

    try:
        agents = client.list_agents()

        if not agents:
            console.print("[yellow]No agents running[/yellow]")
            return

        table = Table(title="Running Agents")
        table.add_column("NAME")
        table.add_column("AGENT ID")
        table.add_column("TEMPLATE")
        table.add_column("STATUS")
        table.add_column("STARTED")

        for agent in agents:
            table.add_row(
                agent.get('name', 'N/A'),
                agent['agent_id'],
                agent['template'],
                agent['status'],
                agent['started_at']
            )

        console.print(table)
    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('agent_id')
@click.argument('message')
def send(agent_id, message):
    """Send a message to an agent"""
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[red]Supervisor is not running[/red]")
        sys.exit(1)

    client = SupervisorClient()

    try:
        result = client.send_message(agent_id, message)
        console.print(f"\n[bold]Agent response:[/bold]\n{result['response']}\n")
    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('agent_id')
def stop(agent_id):
    """Stop an agent"""
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[red]Supervisor is not running[/red]")
        sys.exit(1)

    client = SupervisorClient()

    try:
        client.stop_agent(agent_id)
        console.print(f"[green]Stopped agent {agent_id}[/green]")
    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('agent_id')
def start(agent_id):
    """Start a stopped agent"""
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[red]Supervisor is not running[/red]")
        sys.exit(1)

    client = SupervisorClient()

    try:
        result = client.start_agent(agent_id)
        console.print(f"[green]Started agent {agent_id}[/green]")
    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('agent_id')
def rm(agent_id):
    """Remove an agent completely"""
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[red]Supervisor is not running[/red]")
        sys.exit(1)

    client = SupervisorClient()

    try:
        client.remove_agent(agent_id)
        console.print(f"[green]Removed agent {agent_id}[/green]")
    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('agent_id')
@click.option('--rebuild', is_flag=True, help='Rebuild harness image before upgrade')
@click.option('--refresh-config', is_flag=True, help='Re-read config from template')
def upgrade(agent_id, rebuild, refresh_config):
    """Upgrade an agent's container while preserving identity

    Replaces the container with a new one, preserving memory, Matrix account,
    and chosen name. Use --rebuild to rebuild the harness image first.
    Use --refresh-config to update operational settings from template.

    Examples:
        blue-fairy upgrade researcher-a8f2
        blue-fairy upgrade researcher-a8f2 --rebuild
        blue-fairy upgrade researcher-a8f2 --refresh-config
    """
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[red]Supervisor is not running[/red]")
        sys.exit(1)

    client = SupervisorClient()

    try:
        console.print(f"Upgrading agent {agent_id}...")

        if rebuild:
            console.print("  [dim]Rebuilding harness image...[/dim]")
        if refresh_config:
            console.print("  [dim]Refreshing config from template...[/dim]")

        result = client.upgrade_agent(
            agent_id,
            rebuild=rebuild,
            refresh_config=refresh_config
        )

        name = result.get('name', agent_id)
        console.print(f"[green]✓ Agent '{name}' ({agent_id}) upgraded successfully[/green]")

    except SupervisorError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.group()
def supervisor():
    """Supervisor daemon management"""
    pass


@supervisor.command()
def start():
    """Start supervisor daemon"""
    daemon = DaemonManager()

    if daemon.is_running():
        console.print("[yellow]Supervisor is already running[/yellow]")
        return

    console.print("Starting supervisor...")

    if daemon.start():
        # Wait a moment and verify it started
        import time
        time.sleep(2)

        if daemon.is_running():
            console.print("[green]✓ Supervisor started[/green]")
        else:
            console.print("[red]✗ Supervisor failed to start (check logs)[/red]")
            sys.exit(1)
    else:
        console.print("[red]✗ Failed to start supervisor[/red]")
        sys.exit(1)


@supervisor.command()
def stop():
    """Stop supervisor daemon"""
    daemon = DaemonManager()

    if not daemon.is_running():
        console.print("[yellow]Supervisor is not running[/yellow]")
        return

    console.print("Stopping supervisor...")

    if daemon.stop():
        console.print("[green]✓ Supervisor stopped[/green]")
    else:
        console.print("[red]✗ Failed to stop supervisor[/red]")
        sys.exit(1)


@supervisor.command()
def status():
    """Check supervisor status"""
    daemon = DaemonManager()

    if daemon.is_running():
        console.print("[green]Supervisor is running[/green]")
    else:
        console.print("[yellow]Supervisor is not running[/yellow]")


@supervisor.command()
def logs():
    """View supervisor logs"""
    daemon = DaemonManager()

    if not daemon.log_file.exists():
        console.print("[yellow]No log file found[/yellow]")
        return

    console.print(daemon.log_file.read_text())


@cli.group()
def chat():
    """Matrix chatroom commands"""
    pass


@chat.command()
@click.argument("name")
@click.option("--topic", help="Room topic/description")
@click.option("--invite", multiple=True, help="Matrix user IDs to invite")
def create_room(name: str, topic: str, invite: tuple):
    """Create a new chatroom

    Example:
        blue-fairy chat create-room "Research Team" --topic "AI research" --invite @alice:matrix.org
    """
    import httpx

    payload = {
        "name": name,
        "created_by": "@cli_user:blue-fairy.local"  # TODO: Get actual CLI user ID
    }

    if topic:
        payload["topic"] = topic

    if invite:
        payload["invite"] = list(invite)

    try:
        response = httpx.post(f"{API_BASE}/rooms/create", json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]Created room: {data['room_alias']}[/green]")
        console.print(f"  Room ID: {data['room_id']}")
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to create room: {e}[/red]")
        sys.exit(1)


@chat.command()
@click.argument("room_id")
def delete_room(room_id: str):
    """Delete/archive a chatroom"""
    import httpx

    try:
        response = httpx.delete(f"{API_BASE}/rooms/{room_id}", timeout=30.0)
        response.raise_for_status()

        console.print(f"[green]Deleted room: {room_id}[/green]")
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to delete room: {e}[/red]")
        sys.exit(1)


@chat.command()
@click.option("--all", "show_all", is_flag=True, help="Include archived rooms")
def list_rooms(show_all: bool):
    """List all chatrooms"""
    import httpx

    try:
        response = httpx.get(
            f"{API_BASE}/rooms",
            params={"include_archived": show_all},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        rooms = data["rooms"]

        if not rooms:
            console.print("[yellow]No rooms found[/yellow]")
            return

        console.print(f"\n[bold]Found {len(rooms)} room(s):[/bold]\n")

        for room in rooms:
            status = " [red][ARCHIVED][/red]" if room["is_archived"] else ""
            console.print(f"  [cyan]{room['room_alias']}[/cyan]{status}")
            console.print(f"    Name: {room['room_name']}")
            if room['topic']:
                console.print(f"    Topic: {room['topic']}")
            console.print(f"    ID: {room['room_id']}")
            console.print()

    except httpx.HTTPError as e:
        console.print(f"[red]Failed to list rooms: {e}[/red]")
        sys.exit(1)


@chat.command()
@click.argument("room_id")
@click.argument("user_or_agent")
def invite(room_id: str, user_or_agent: str):
    """Invite a user or agent to a room

    Auto-detects whether the identifier is a Matrix user ID (@user:server) or
    an agent ID (template-xxxx). Matrix user IDs must start with @.

    Examples:
        blue-fairy chat invite "#research:blue-fairy.local" @alice:matrix.org
        blue-fairy chat invite "#research:blue-fairy.local" researcher-a8f2
    """
    import httpx

    # Auto-detect: Matrix user IDs start with @, everything else is an agent ID
    is_matrix_user = user_or_agent.startswith("@")

    try:
        if is_matrix_user:
            # Invite by user_id (Matrix user)
            response = httpx.post(
                f"{API_BASE}/rooms/invite",
                json={"room_id": room_id, "user_id": user_or_agent},
                timeout=30.0
            )
            response.raise_for_status()
            console.print(f"[green]Invited user {user_or_agent} to {room_id}[/green]")
        else:
            # Invite by agent_id
            response = httpx.post(
                f"{API_BASE}/rooms/invite",
                json={"room_id": room_id, "agent_id": user_or_agent},
                timeout=30.0
            )
            response.raise_for_status()
            console.print(f"[green]Invited agent {user_or_agent} to {room_id}[/green]")
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to invite: {e}[/red]")
        sys.exit(1)


@chat.command()
@click.argument("room_id")
@click.argument("user_id")
def remove(room_id: str, user_id: str):
    """Remove a user from a room

    Example:
        blue-fairy chat remove "#research:blue-fairy.local" @alice:matrix.org
    """
    import httpx

    try:
        response = httpx.post(
            f"{API_BASE}/rooms/{room_id}/remove",
            json={"user_id": user_id},
            timeout=10.0
        )
        response.raise_for_status()

        console.print(f"[green]Removed {user_id} from {room_id}[/green]")
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to remove user: {e}[/red]")
        sys.exit(1)


@chat.command()
@click.argument("room_id")
def members(room_id: str):
    """List members of a room"""
    import httpx

    try:
        response = httpx.get(f"{API_BASE}/rooms/{room_id}/members", timeout=10.0)
        response.raise_for_status()
        data = response.json()

        members = data["members"]

        if not members:
            console.print("[yellow]No members found[/yellow]")
            return

        console.print(f"\n[bold]Room has {len(members)} member(s):[/bold]\n")

        for member in members:
            console.print(f"  [cyan]{member['user_id']}[/cyan] ({member['member_type']})")
            if member.get('agent_id'):
                console.print(f"    Agent ID: {member['agent_id']}")

    except httpx.HTTPError as e:
        console.print(f"[red]Failed to get members: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()

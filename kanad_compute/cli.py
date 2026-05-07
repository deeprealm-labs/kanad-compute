"""Kanad Compute CLI — Terminal interface for setup and server management."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

BANNER = """[bold cyan]
 █▄▀  ▄▀█  █▄  █  ▄▀█  █▀▄
 █ █  █▀█  █ ▀▄█  █▀█  █▄▀
[/bold cyan][dim] compute · quantum chemistry on your machine[/dim]"""


@click.group()
def main():
    """Kanad Compute — Turn your computer into a quantum chemistry server."""
    pass


@main.command()
@click.option("--port", default=7440, help="Server port (default: 7440)")
@click.option("--max-qubits", default=20, help="Max qubits to accept (default: 20)")
@click.option("--gpu/--no-gpu", default=False, help="Enable GPU acceleration")
@click.option("--ibm-token", default=None, help="IBM Quantum API token")
@click.option("--ionq-key", default=None, help="IonQ API key")
@click.option("--ngrok-token", default=None, help="ngrok auth token for public tunnels")
def init(port, max_qubits, gpu, ibm_token, ionq_key, ngrok_token):
    """Initialize Kanad Compute configuration."""
    from .config import init_config

    console.print(BANNER)
    console.print()

    cfg = init_config(
        port=port, max_qubits=max_qubits, gpu=gpu,
        ibm_token=ibm_token, ionq_key=ionq_key,
    )
    if ngrok_token:
        cfg["ngrok_token"] = ngrok_token
        from .config import save_config
        save_config(cfg)

    # Show config
    table = Table(title="Configuration", box=box.SIMPLE_HEAVY, title_style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Node ID", cfg["node_id"][:12] + "...")
    table.add_row("API Key", cfg["api_key"][:12] + "...")
    table.add_row("Port", str(cfg["port"]))
    table.add_row("Max Qubits", str(cfg["max_qubits"]))
    table.add_row("GPU", "Enabled" if cfg["gpu_enabled"] else "Disabled")
    table.add_row("IBM Quantum", "Configured" if cfg["ibm_api_token"] else "Not set")
    table.add_row("IonQ", "Configured" if cfg["ionq_api_key"] else "Not set")

    console.print(table)
    console.print()

    # Show the key to paste into Kanad
    console.print(Panel(
        f"[bold green]Your Kanad Compute API Key:[/bold green]\n\n"
        f"[bold white]{cfg['api_key']}[/bold white]\n\n"
        f"[dim]1. Paste this key in Kanad → Profile → Backend Credentials → Compute Key[/dim]\n"
        f"[dim]2. Start the server:[/dim]\n"
        f"[dim]   kanad-compute start --connect https://kanad-api-640826962316.us-central1.run.app[/dim]\n"
        f"[dim]   (or use http://localhost:8000 for local dev)[/dim]",
        title="[bold]Ready to connect[/bold]",
        border_style="green",
    ))

    from .config import CONFIG_FILE
    console.print(f"\n[dim]Config saved to: {CONFIG_FILE}[/dim]")
    console.print(f"[dim]Run [bold]kanad-compute start[/bold] to begin serving.[/dim]\n")


@main.command()
@click.option("--host", default=None, help="Override host (default: 0.0.0.0)")
@click.option("--port", default=None, type=int, help="Override port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development)")
@click.option("--connect", default=None, help="Connect to Kanad platform (e.g. https://kanad-api-640826962316.us-central1.run.app or http://localhost:8000)")
def start(host, port, reload, connect):
    """Start the Kanad Compute server."""
    from .config import load_config, save_config, CONFIG_FILE

    if not CONFIG_FILE.exists():
        console.print("[red]No config found. Run [bold]kanad-compute init[/bold] first.[/red]")
        raise SystemExit(1)

    cfg = load_config()
    _host = host or cfg.get("host", "0.0.0.0")
    _port = port or cfg.get("port", 7440)

    console.print(BANNER)
    console.print()

    # System info
    from .sysinfo import get_system_info
    info = get_system_info(cfg.get("gpu_enabled", False))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column(style="dim")
    table.add_column(style="white")
    table.add_row("CPU", f"{info['cpu_physical']} cores ({info['cpu_count']} threads)")
    table.add_row("RAM", f"{info['ram_available_gb']} / {info['ram_total_gb']} GB")
    if info.get("gpu_available"):
        table.add_row("GPU", f"{info['gpu_name']} ({info['gpu_memory_gb']} GB)")
    table.add_row("Max Qubits", str(cfg.get("max_qubits", 20)))
    table.add_row("Node ID", cfg["node_id"][:12] + "...")

    console.print(Panel(table, title="[bold]System[/bold]", border_style="blue"))
    console.print(f"\n  Local:  [bold green]http://{_host}:{_port}[/bold green]")
    console.print(f"  API Key: [dim]{cfg['api_key'][:16]}...[/dim]")

    # Connect to Kanad platform (outbound — no port forwarding needed)
    kanad_url = connect or cfg.get("kanad_url")
    if kanad_url:
        kanad_url = kanad_url.rstrip('/')
        console.print(f"  Kanad:  [bold cyan]{kanad_url}[/bold cyan]")
        console.print()
        console.print(Panel(
            f"[bold]Connected to Kanad platform[/bold]\n\n"
            f"  [cyan]{kanad_url}[/cyan]\n\n"
            f"[dim]Your computer will receive and run quantum chemistry jobs from kanad.xyz[/dim]\n"
            f"[dim]No port forwarding or tunnels needed — your machine connects outbound.[/dim]",
            title="[bold green]Worker Mode[/bold green]",
            border_style="green",
        ))
        cfg["kanad_url"] = kanad_url
        save_config(cfg)
    else:
        console.print()
        console.print("  [dim]Local mode only. To connect to kanad.xyz:[/dim]")
        console.print("  [dim]  kanad-compute start --connect https://kanad-api-640826962316.us-central1.run.app[/dim]")

    console.print(f"\n  Press [bold]Ctrl+C[/bold] to stop.\n")

    import uvicorn
    from .server import create_app

    app = create_app(cfg)
    app.state.public_url = None

    # Start worker thread if connected to Kanad
    worker_thread = None
    if kanad_url:
        import threading
        from .remote_worker import start_worker
        worker_thread = threading.Thread(
            target=start_worker,
            args=(kanad_url, cfg),
            daemon=True,
        )
        worker_thread.start()

    uvicorn.run(
        app, host=_host, port=_port,
        log_level=cfg.get("log_level", "info"),
        reload=reload,
    )


@main.command()
def status():
    """Check server status and system info."""
    from .config import load_config, CONFIG_FILE
    from .sysinfo import get_system_info

    console.print(BANNER)
    console.print()

    if not CONFIG_FILE.exists():
        console.print("[yellow]Not initialized. Run [bold]kanad-compute init[/bold] first.[/yellow]")
        return

    cfg = load_config()
    info = get_system_info(cfg.get("gpu_enabled", False))

    # System
    table = Table(title="System Info", box=box.ROUNDED)
    table.add_column("", style="cyan")
    table.add_column("", style="white")
    table.add_row("Platform", f"{info['platform']} {info['arch']}")
    table.add_row("Python", info["python"])
    table.add_row("CPU", f"{info['cpu_physical']} cores / {info['cpu_count']} threads")
    table.add_row("RAM", f"{info['ram_total_gb']} GB total, {info['ram_available_gb']} GB free")
    table.add_row("GPU", info.get("gpu_name") or "Not detected")
    table.add_row("CUDA", "Available" if info.get("cuda_available") else "Not available")
    console.print(table)

    # Packages
    pkg_table = Table(title="Quantum Packages", box=box.ROUNDED)
    pkg_table.add_column("Package", style="cyan")
    pkg_table.add_column("Version", style="white")
    for pkg, ver in info.get("packages", {}).items():
        style = "green" if ver else "dim red"
        pkg_table.add_row(pkg, ver or "not installed", style=style)
    console.print(pkg_table)

    # Config
    cfg_table = Table(title="Configuration", box=box.ROUNDED)
    cfg_table.add_column("Setting", style="cyan")
    cfg_table.add_column("Value", style="white")
    cfg_table.add_row("Node ID", cfg["node_id"][:12] + "...")
    cfg_table.add_row("Port", str(cfg["port"]))
    cfg_table.add_row("Max Qubits", str(cfg["max_qubits"]))
    cfg_table.add_row("GPU Mode", "Enabled" if cfg.get("gpu_enabled") else "Disabled")
    cfg_table.add_row("IBM Quantum", "Configured" if cfg.get("ibm_api_token") else "Not set")
    cfg_table.add_row("IonQ", "Configured" if cfg.get("ionq_api_key") else "Not set")
    console.print(cfg_table)

    # Check if server is running
    import httpx
    port = cfg.get("port", 7440)
    try:
        r = httpx.get(f"http://localhost:{port}/health", timeout=2)
        if r.status_code == 200:
            console.print(f"\n[bold green]Server is RUNNING on port {port}[/bold green]\n")
        else:
            console.print(f"\n[yellow]Server responded with {r.status_code}[/yellow]\n")
    except Exception:
        console.print(f"\n[dim]Server is not running on port {port}[/dim]")
        console.print(f"[dim]Start with: [bold]kanad-compute start[/bold][/dim]\n")


@main.command()
@click.option("--url", default=None, help="Kanad URL (default: configured kanad_url)")
def connect(url):
    """Connect to Kanad over WebSocket (replaces polling worker)."""
    from .config import load_config, save_config, CONFIG_FILE

    if not CONFIG_FILE.exists():
        console.print("[red]Not initialized. Run [bold]kanad-compute init[/bold] first.[/red]")
        raise SystemExit(1)

    cfg = load_config()
    kanad_url = url or cfg.get("kanad_url") or cfg.get("kanad_api_url")
    if not kanad_url:
        console.print("[red]No Kanad URL set. Use --url or run [bold]kanad-compute start --connect URL[/bold] first.[/red]")
        raise SystemExit(1)

    cfg["kanad_url"] = kanad_url.rstrip("/")
    save_config(cfg)

    console.print(BANNER)
    console.print()
    console.print(Panel(
        f"[bold]Connecting to Kanad over WebSocket[/bold]\n\n"
        f"  [cyan]{kanad_url}[/cyan]\n\n"
        f"[dim]Persistent outbound connection. Jobs are pushed live; results stream back.[/dim]",
        title="[bold green]WS Mode[/bold green]",
        border_style="green",
    ))
    console.print(f"\n  Press [bold]Ctrl+C[/bold] to disconnect.\n")

    from .ws_client import ComputeWSClient
    client = ComputeWSClient(kanad_url, cfg)
    try:
        client.run_forever_sync()
    except KeyboardInterrupt:
        console.print("\n[dim]Disconnected.[/dim]")


@main.command()
def key():
    """Display the API key (for pasting into Kanad app)."""
    from .config import load_config, CONFIG_FILE

    if not CONFIG_FILE.exists():
        console.print("[red]Not initialized. Run [bold]kanad-compute init[/bold] first.[/red]")
        raise SystemExit(1)

    cfg = load_config()
    console.print(f"\n[bold]{cfg['api_key']}[/bold]\n")


@main.command()
@click.option("--ibm-token", default=None, help="Set IBM Quantum API token")
@click.option("--ionq-key", default=None, help="Set IonQ API key")
@click.option("--max-qubits", default=None, type=int, help="Set max qubits")
@click.option("--gpu/--no-gpu", default=None, help="Enable/disable GPU")
@click.option("--port", default=None, type=int, help="Set port")
def configure(ibm_token, ionq_key, max_qubits, gpu, port):
    """Update configuration settings."""
    from .config import load_config, save_config, CONFIG_FILE

    if not CONFIG_FILE.exists():
        console.print("[red]Not initialized. Run [bold]kanad-compute init[/bold] first.[/red]")
        raise SystemExit(1)

    cfg = load_config()
    changed = []

    if ibm_token is not None:
        cfg["ibm_api_token"] = ibm_token or None
        changed.append("IBM token")
    if ionq_key is not None:
        cfg["ionq_api_key"] = ionq_key or None
        changed.append("IonQ key")
    if max_qubits is not None:
        cfg["max_qubits"] = max_qubits
        changed.append("max_qubits")
    if gpu is not None:
        cfg["gpu_enabled"] = gpu
        changed.append("GPU mode")
    if port is not None:
        cfg["port"] = port
        changed.append("port")

    if changed:
        save_config(cfg)
        console.print(f"[green]Updated: {', '.join(changed)}[/green]")
    else:
        console.print("[dim]No changes. Use --help to see options.[/dim]")


@main.group()
def creds():
    """Manage local credential vault (OS keyring)."""
    pass


_VAULT_CHOICES = ("ibm_api_token", "ibm_crn", "ionq_api_key", "bluequbit_api_key")


@creds.command("set")
@click.argument("key", type=click.Choice(_VAULT_CHOICES))
@click.option("--value", default=None, help="Value (prompted if omitted)")
def creds_set(key, value):
    """Store a credential in the OS keyring."""
    from .vault import Vault, VaultError
    if value is None:
        value = click.prompt(f"Enter value for {key}", hide_input=True, confirmation_prompt=False)
    try:
        Vault().set(key, value)
        console.print(f"[green]✓ stored {key}[/green]")
    except VaultError as e:
        console.print(f"[red]vault error: {e}[/red]")
        raise SystemExit(1)


@creds.command("get")
@click.argument("key", type=click.Choice(_VAULT_CHOICES))
@click.option("--reveal", is_flag=True, help="Print the full secret (default: last 4 chars only)")
def creds_get(key, reveal):
    """Read a credential from the vault. Hidden by default."""
    from .vault import Vault
    val = Vault().get(key)
    if val is None:
        console.print(f"[yellow]{key}: not set[/yellow]")
        return
    if reveal:
        console.print(val)
    else:
        console.print(f"{key}: ****{val[-4:] if len(val) >= 4 else '****'}")


@creds.command("list")
def creds_list():
    """List which logical credentials are present."""
    from .vault import Vault
    status = Vault().status()
    table = Table(box=box.SIMPLE)
    table.add_column("backend", style="cyan")
    table.add_column("present")
    for backend, ok in status.items():
        table.add_row(backend, "[green]yes[/green]" if ok else "[dim]no[/dim]")
    console.print(table)


@creds.command("clear")
@click.argument("key", type=click.Choice(_VAULT_CHOICES))
def creds_clear(key):
    """Remove a credential from the vault."""
    from .vault import Vault
    if Vault().clear(key):
        console.print(f"[green]✓ cleared {key}[/green]")
    else:
        console.print(f"[yellow]{key}: nothing to clear[/yellow]")


if __name__ == "__main__":
    main()

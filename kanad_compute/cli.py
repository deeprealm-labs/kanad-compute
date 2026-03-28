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


if __name__ == "__main__":
    main()

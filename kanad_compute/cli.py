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
        f"[dim]Paste this in Kanad → Profile → Backend Credentials → Compute Key[/dim]\n"
        f"[dim]When you run [bold]kanad-compute start[/bold], a public URL will be generated.[/dim]\n"
        f"[dim]Paste that URL in Kanad → Profile → Backend Credentials → Compute URL[/dim]",
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
@click.option("--public/--no-public", default=True, help="Expose via public tunnel (default: yes)")
@click.option("--tunnel", default="ssh", type=click.Choice(["ssh", "ngrok"]), help="Tunnel method (default: ssh — no signup needed)")
@click.option("--ngrok-token", default=None, help="ngrok auth token (only for --tunnel ngrok)")
def start(host, port, reload, public, tunnel, ngrok_token):
    """Start the Kanad Compute server."""
    import subprocess
    import threading
    import re
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

    # Start public tunnel
    public_url = None
    tunnel_proc = None

    if public:
        if tunnel == "ssh":
            # SSH tunnel via localhost.run — zero signup, zero config
            public_url = _start_ssh_tunnel(_port, cfg, save_config)
        elif tunnel == "ngrok":
            public_url = _start_ngrok_tunnel(_port, cfg, save_config, ngrok_token)

        if public_url:
            console.print(f"  Public: [bold cyan]{public_url}[/bold cyan]")
            console.print()
            console.print(Panel(
                f"[bold]Paste this URL in Kanad → Profile → Compute URL:[/bold]\n\n"
                f"  [bold cyan]{public_url}[/bold cyan]\n\n"
                f"[dim]Accessible from kanad.xyz and anywhere on the internet.[/dim]",
                title="[bold green]Public URL[/bold green]",
                border_style="green",
            ))
        else:
            console.print("  [yellow]Tunnel failed. Server only accessible locally.[/yellow]")
    else:
        console.print("  [dim]Public tunnel disabled. Use --public to expose.[/dim]")

    console.print(f"\n  Press [bold]Ctrl+C[/bold] to stop.\n")

    import uvicorn
    from .server import create_app

    app = create_app(cfg)
    app.state.public_url = public_url

    try:
        uvicorn.run(
            app, host=_host, port=_port,
            log_level=cfg.get("log_level", "info"),
            reload=reload,
        )
    finally:
        pass


def _start_ssh_tunnel(port: int, cfg: dict, save_fn) -> str | None:
    """Start SSH tunnel via localhost.run (no signup needed)."""
    import subprocess
    import threading
    import re
    import time

    public_url = None
    url_event = threading.Event()

    def _run_tunnel():
        nonlocal public_url
        try:
            proc = subprocess.Popen(
                ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
                 "-R", f"80:localhost:{port}", "localhost.run"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="ignore").strip()
                # localhost.run outputs: https://XXXXX.lhr.life
                match = re.search(r'(https?://\S+\.lhr\.life\S*)', text)
                if not match:
                    match = re.search(r'(https?://\S+localhost\.run\S*)', text)
                if match:
                    public_url = match.group(1).rstrip(',').rstrip()
                    cfg["public_url"] = public_url
                    save_fn(cfg)
                    url_event.set()
        except Exception as e:
            console.print(f"  [yellow]SSH tunnel error: {e}[/yellow]")
            url_event.set()

    t = threading.Thread(target=_run_tunnel, daemon=True)
    t.start()

    # Wait up to 15s for the URL
    url_event.wait(timeout=15)
    if not public_url:
        # Try serveo.net as fallback
        return _start_serveo_tunnel(port, cfg, save_fn)
    return public_url


def _start_serveo_tunnel(port: int, cfg: dict, save_fn) -> str | None:
    """Fallback: SSH tunnel via serveo.net."""
    import subprocess
    import threading
    import re

    public_url = None
    url_event = threading.Event()

    def _run():
        nonlocal public_url
        try:
            proc = subprocess.Popen(
                ["ssh", "-tt", "-o", "StrictHostKeyChecking=no",
                 "-R", f"0:localhost:{port}", "serveo.net"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="ignore").strip()
                match = re.search(r'(https?://\S+serveo\.net\S*)', text)
                if match:
                    public_url = match.group(1).rstrip(',').rstrip()
                    cfg["public_url"] = public_url
                    save_fn(cfg)
                    url_event.set()
        except Exception:
            url_event.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    url_event.wait(timeout=15)
    return public_url


def _start_ngrok_tunnel(port: int, cfg: dict, save_fn, token: str | None) -> str | None:
    """Start ngrok tunnel (requires token)."""
    try:
        from pyngrok import ngrok, conf
        auth = token or cfg.get("ngrok_token")
        if not auth:
            console.print("  [yellow]ngrok requires auth token. Get one at https://ngrok.com[/yellow]")
            console.print("  [dim]  kanad-compute start --ngrok-token YOUR_TOKEN[/dim]")
            console.print("  [dim]Or use SSH tunnel (default): kanad-compute start --tunnel ssh[/dim]")
            return None
        conf.get_default().auth_token = auth
        tun = ngrok.connect(port, "http")
        public_url = tun.public_url
        cfg["public_url"] = public_url
        save_fn(cfg)
        return public_url
    except Exception as e:
        console.print(f"  [yellow]ngrok failed: {e}[/yellow]")
        return None


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

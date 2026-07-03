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
@click.option("--max-qubits", default=33, help="Max qubits to accept (default: 33)")
@click.option("--gpu/--no-gpu", default=False, help="Enable GPU acceleration")
@click.option("--gpu-device", default="auto", type=click.Choice(["auto", "amd", "nvidia", "cpu"]),
              help="GPU engine: amd (rocm-planck) | nvidia (cudaq) | cpu | auto")
@click.option("--ibm-token", default=None, help="IBM Quantum API token")
@click.option("--ionq-key", default=None, help="IonQ API key")
@click.option("--ngrok-token", default=None, help="ngrok auth token for public tunnels")
def init(port, max_qubits, gpu, gpu_device, ibm_token, ionq_key, ngrok_token):
    """Initialize Kanad Compute configuration."""
    from .config import init_config

    console.print(BANNER)
    console.print()

    cfg = init_config(
        port=port, max_qubits=max_qubits, gpu=gpu, gpu_device=gpu_device,
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
@click.option("--host", default=None, help="Override host (default from config: 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Override port")
@click.option("--reload", is_flag=True, help="Auto-reload (development; implies server-only)")
@click.option("--no-tui", is_flag=True, help="Server only, no live dashboard (headless/systemd)")
@click.option("--force-qpu", is_flag=True,
              help="For SQD/IBM jobs, run ONLY on the QPU — fail with the real error instead of "
                   "silently falling back to local statevector sampling.")
def start(host, port, reload, no_tui, force_qpu):
    """Start the Kanad Compute node. Shows a live dashboard in a terminal; runs
    server-only when headless (--no-tui / no TTY). Polling nodes dial OUT to kanad-app;
    SSH nodes bind localhost."""
    import sys
    import threading
    import time
    from .config import load_config, CONFIG_FILE

    if not CONFIG_FILE.exists():
        console.print("[red]No config found. Run [bold]kanad-compute init[/bold] first.[/red]")
        raise SystemExit(1)

    cfg = load_config()
    # --force-qpu (or persisted force_qpu in config) → no statevector fallback for IBM SQD.
    if force_qpu:
        cfg["force_qpu"] = True

    # Zero-config polling node: dial OUT to kanad-app, pull jobs, push results.
    # No SSH, no inbound, works behind any NAT. Set by `kanad-compute connect`.
    if cfg.get("transport") == "polling":
        _start_polling(cfg)
        return

    _host = host or cfg.get("host", "127.0.0.1")
    _port = port or cfg.get("port", 7440)

    import uvicorn
    from .server import create_app
    app = create_app(cfg)
    app.state.public_url = None

    # Headless / no terminal -> server only (blocking), no dashboard.
    if no_tui or reload or not sys.stdout.isatty():
        console.print(BANNER)
        console.print(f"\n  Serving [bold green]http://{_host}:{_port}[/bold green]  ·  node {cfg['node_id'][:8]}  ·  gpu {cfg.get('gpu_device','auto')}")
        console.print("  [dim]Reached by kanad-app over SSH. Ctrl+C to stop.[/dim]\n")
        uvicorn.run(app, host=_host, port=_port,
                    log_level=cfg.get("log_level", "info"), reload=reload)
        return

    # Interactive -> uvicorn in a daemon thread + live TUI in the main thread.
    sconf = uvicorn.Config(app, host=_host, port=_port, log_level="critical")
    server = uvicorn.Server(sconf)
    server.install_signal_handlers = lambda: None   # not the main thread
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    try:
        _run_tui(cfg, _port)
    finally:
        server.should_exit = True
        time.sleep(0.3)


def _run_tui(cfg, port):
    """Live node dashboard — polls the node's own local API (decoupled from the
    server internals) and renders a rich.Live layout: connection/pairing, the
    pushed login session, the compute engine, and jobs."""
    import time
    import getpass
    import httpx
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text

    base = f"http://127.0.0.1:{port}"
    hdr = {"Authorization": f"Bearer {cfg['api_key']}"}
    ip = _public_ip()
    user = getpass.getuser()
    try:
        info0 = httpx.get(base + "/info", headers=hdr, timeout=4).json()   # static: poll once
    except Exception:
        info0 = {}

    def _poll(path):
        try:
            return httpx.get(base + path, headers=hdr, timeout=3).json()
        except Exception:
            return None

    def render():
        sess = _poll("/session") or {}
        jobs = _poll("/jobs") or []
        paired = bool(sess.get("user_email") or sess.get("email"))

        lay = Layout()
        lay.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3))
        lay["body"].split_row(Layout(name="left"), Layout(name="right"))
        lay["left"].split_column(Layout(name="conn"), Layout(name="session"))
        lay["right"].split_column(Layout(name="compute"), Layout(name="jobs"))

        lay["header"].update(Panel(Text("KANAD COMPUTE   ·   quantum chemistry node", justify="center", style="bold cyan"),
                                   subtitle=f"node {cfg['node_id'][:8]} · [green]● serving[/green]"))

        c = Table(box=box.SIMPLE, show_header=False); c.add_column(style="dim"); c.add_column(style="white")
        c.add_row("Public IP", ip)
        c.add_row("SSH", f"{user}@{ip}:22")
        c.add_row("API key", cfg["api_key"][:18] + "…")
        c.add_row("Pairing", "[green]✓ paired[/green]" if paired
                  else "[yellow]⧗ waiting — run `kanad-compute authorize`, then add this node in the app[/yellow]")
        lay["conn"].update(Panel(c, title="Connection", border_style="blue"))

        if paired:
            s = Table(box=box.SIMPLE, show_header=False); s.add_column(style="dim"); s.add_column(style="white")
            s.add_row("User", str(sess.get("user_email") or sess.get("email")))
            s.add_row("Plan", str(sess.get("plan") or "—"))
            s.add_row("Compute runs", str(sess.get("runs", 0)))
            exps = sess.get("experiments") or []
            s.add_row("Experiments", str(len(exps)))
            for e in exps[:6]:
                en = e.get("energy")
                ens = f"{en:.4f}" if isinstance(en, (int, float)) else ""
                s.add_row("", f"[dim]{e.get('type') or '?'}  {e.get('status') or ''}  {ens}[/dim]")
            sched = sess.get("scheduled") or []
            if sched:
                s.add_row("Scheduled", str(len(sched)))
            lay["session"].update(Panel(s, title="Session", border_style="green"))
        else:
            lay["session"].update(Panel(
                Text("Not paired yet.\n\nIn kanad-app → Backend Config → Add node,\nenter the IP / SSH user / API key shown left.\n\nThe app then pushes your login here.", style="dim"),
                title="Session", border_style="grey50"))

        m = Table(box=box.SIMPLE, show_header=False); m.add_column(style="dim"); m.add_column(style="white")
        eng = info0.get("gpu_engine", "cpu")
        m.add_row("Engine", f"[bold]{eng}[/bold]" + (f"  ({info0.get('gpu_name')})" if info0.get("gpu_name") else ""))
        m.add_row("GPU vendor", info0.get("gpu_vendor") or "none")
        m.add_row("Max qubits", str(info0.get("max_qubits", cfg.get("max_qubits", 33))))
        m.add_row("CPU / RAM", f"{info0.get('cpu_physical', '?')}c · {info0.get('ram_total_gb', '?')} GB")
        active = sum(1 for j in (jobs or []) if j.get("status") in ("pending", "running"))
        m.add_row("Active jobs", f"{active} / {info0.get('max_workers', cfg.get('max_workers', 2))}")
        lay["compute"].update(Panel(m, title="Compute", border_style="magenta"))

        j = Table(box=box.SIMPLE); j.add_column("job", style="dim"); j.add_column("status")
        for job in (jobs or [])[:8]:
            st = job.get("status", "?")
            color = ("green" if st == "completed" else "yellow" if st in ("running", "pending")
                     else "red" if st in ("failed", "error", "cancelled") else "white")
            j.add_row(str(job.get("job_id", ""))[:8], f"[{color}]{st}[/{color}]")
        if not jobs:
            j.add_row("[dim]—[/dim]", "[dim]no jobs yet[/dim]")
        lay["jobs"].update(Panel(j, title="Jobs", border_style="cyan"))

        lay["footer"].update(Panel(Text("Ctrl+C to stop   ·   reached by kanad-app over SSH (no public HTTP port)",
                                        justify="center", style="dim")))
        return lay

    with Live(render(), console=console, refresh_per_second=4, screen=True) as live:
        try:
            while True:
                time.sleep(2)
                live.update(render())
        except KeyboardInterrupt:
            pass


def _start_polling(cfg):
    """Run as an outbound polling node: register with kanad-app, poll for jobs,
    execute locally, push results. No inbound connection required."""
    from .remote_worker import start_worker

    kanad_url = (cfg.get("kanad_api_url") or cfg.get("kanad_url") or "").rstrip("/")
    if not kanad_url:
        console.print("[red]No kanad_api_url in config. Re-run [bold]kanad-compute connect <token>[/bold].[/red]")
        raise SystemExit(1)
    if not cfg.get("api_key"):
        console.print("[red]Not connected. Run [bold]kanad-compute connect <token>[/bold] first.[/red]")
        raise SystemExit(1)

    console.print(BANNER)
    console.print(f"\n  Connecting to [bold green]{kanad_url}[/bold green] as a polling node")
    console.print(f"  node [bold white]{cfg.get('node_id','?')[:8]}[/bold white] · engine {cfg.get('gpu_device','auto')} · max qubits {cfg.get('max_qubits', 33)}")
    console.print("  [dim]Dialing out — no inbound connection, SSH or public port needed. Ctrl+C to stop.[/dim]\n")
    import sys
    import threading
    import time as _time
    from .sysinfo import get_system_info

    # Headless (systemd / docker / no TTY): plain blocking loop, no dashboard.
    if not sys.stdout.isatty():
        try:
            start_worker(kanad_url, cfg)
        except KeyboardInterrupt:
            pass
        return

    # Interactive: worker in a daemon thread + a live dashboard in the main thread.
    status = {"connected": False, "polls": 0, "active": 0, "recent": [], "last_error": None}
    info = {}
    try:
        info = get_system_info(cfg.get("gpu_enabled", False))
    except Exception:
        pass
    threading.Thread(target=start_worker, args=(kanad_url, cfg, status), daemon=True).start()

    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text

    def render():
        lay = Layout()
        lay.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3))
        lay["body"].split_row(Layout(name="left"), Layout(name="right"))
        conn = "[green]connected[/green]" if status.get("connected") else "[yellow]connecting...[/yellow]"
        lay["header"].update(Panel(Text("KANAD COMPUTE   .   polling node", justify="center", style="bold cyan"),
                                   subtitle=f"node {cfg.get('node_id','?')[:8]} . {conn}"))
        c = Table(box=box.SIMPLE, show_header=False); c.add_column(style="dim"); c.add_column(style="white")
        c.add_row("Platform", kanad_url.replace("https://", ""))
        c.add_row("Transport", "polling (dials out)")
        c.add_row("Status", conn)
        c.add_row("Polls", str(status.get("polls", 0)))
        eng = info.get("gpu_engine", "cpu")
        c.add_row("Engine", f"[bold]{eng}[/bold]" + (f" ({info.get('gpu_name')})" if info.get("gpu_name") else ""))
        c.add_row("Max qubits", str(info.get("max_qubits", cfg.get("max_qubits", 33))))
        c.add_row("CPU / RAM", f"{info.get('cpu_physical','?')}c . {info.get('ram_total_gb','?')} GB")
        c.add_row("Active jobs", str(status.get("active", 0)))
        if status.get("last_error"):
            c.add_row("Last error", f"[red]{str(status['last_error'])[:38]}[/red]")
        lay["left"].update(Panel(c, title="Node", border_style="blue"))
        j = Table(box=box.SIMPLE); j.add_column("job", style="dim"); j.add_column("system"); j.add_column("solver", style="dim"); j.add_column("status")
        for e in status.get("recent", [])[:8]:
            st = e.get("status", "?")
            color = "green" if st == "completed" else "yellow" if st == "running" else "red"
            en = e.get("energy"); ens = f"  {en:.4f} Ha" if isinstance(en, (int, float)) else ""
            j.add_row(str(e.get("id",""))[:8], str(e.get("name","?"))[:14], str(e.get("solver","?"))[:14], f"[{color}]{st}{ens}[/{color}]")
        if not status.get("recent"):
            j.add_row("[dim]--[/dim]", "[dim]waiting for jobs[/dim]", "", "")
        lay["right"].update(Panel(j, title="Jobs", border_style="cyan"))
        lay["footer"].update(Panel(Text("Dialing out over HTTPS -- no SSH, public port or inbound needed.   Ctrl+C to stop.",
                                        justify="center", style="dim")))
        return lay

    with Live(render(), console=console, refresh_per_second=4, screen=True) as live:
        try:
            while True:
                _time.sleep(0.5)
                live.update(render())
        except KeyboardInterrupt:
            pass


@main.command()
@click.argument("token")
@click.option("--api-url", default=None, help="kanad-app API base (default: config / production)")
def connect(token, api_url):
    """Connect this machine to kanad-app as a zero-config polling node.

    Paste the token from kanad-app → Backend Config → Connect a node. The node
    dials OUT over HTTPS — no SSH, no public IP, no port-forwarding. Then run
    `kanad-compute start`.
    """
    from .config import load_config, save_config, CONFIG_FILE, _default_config

    token = (token or "").strip()
    if not token:
        console.print("[red]Empty token. Copy it from kanad-app → Backend Config → Connect a node.[/red]")
        raise SystemExit(1)

    cfg = load_config() if CONFIG_FILE.exists() else _default_config()
    cfg["api_key"] = token          # the enrollment token IS the node's bearer
    cfg["transport"] = "polling"
    if api_url:
        cfg["kanad_api_url"] = api_url.rstrip("/")
    cfg.setdefault("kanad_api_url", "https://kanad-api-640826962316.us-central1.run.app")
    save_config(cfg)

    console.print(BANNER)
    console.print(f"\n  [green]✓ Connected[/green] to [bold]{cfg['kanad_api_url']}[/bold]")
    console.print(f"  node [bold white]{cfg.get('node_id','?')[:8]}[/bold white]")
    console.print("\n  Now run [bold cyan]kanad-compute start[/bold cyan] to begin taking jobs.\n")


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


def _public_ip() -> str:
    import httpx
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"):
        try:
            ip = httpx.get(url, timeout=5).text.strip()
            if ip:
                return ip
        except Exception:
            continue
    return "<your-public-ip>"


@main.command()
def pair():
    """Show the details to pair this node in kanad-app."""
    from .config import load_config, CONFIG_FILE
    import getpass

    if not CONFIG_FILE.exists():
        console.print("[red]Not initialized. Run [bold]kanad-compute init[/bold] first.[/red]")
        raise SystemExit(1)

    cfg = load_config()
    console.print(BANNER)
    console.print()
    ip = _public_ip()
    user = getpass.getuser()
    console.print(Panel(
        f"[bold]Pair in kanad-app → Profile → Backend Config → Compute Nodes → Add node[/bold]\n\n"
        f"  SSH host      [bold white]{ip}[/bold white]\n"
        f"  SSH port      [bold white]22[/bold white]\n"
        f"  SSH user      [bold white]{user}[/bold white]\n"
        f"  Node API key  [bold green]{cfg['api_key']}[/bold green]\n"
        f"  Node API port [bold white]{cfg.get('port', 7440)}[/bold white]\n\n"
        f"[dim]First run [bold]kanad-compute authorize[/bold] so kanad-app may connect over SSH,[/dim]\n"
        f"[dim]then [bold]kanad-compute start[/bold] to keep the node online.[/dim]",
        title="[bold cyan]Pairing details[/bold cyan]",
        border_style="cyan",
    ))


@main.command()
@click.option("--api-url", default=None, help="kanad-app API base (default: config kanad_api_url)")
@click.option("--key", "pubkey", default=None, help="Authorize this explicit public key instead of fetching")
def authorize(api_url, pubkey):
    """Authorize kanad-app's SSH key so it can connect to this node.

    Fetches kanad-app's platform public key and adds it to ~/.ssh/authorized_keys,
    locked down so it may ONLY port-forward to this node's local compute API
    (no shell, no other forwards).
    """
    from .config import load_config
    import os
    import httpx

    cfg = load_config()
    port = cfg.get("port", 7440)

    if not pubkey:
        base = (api_url or cfg.get("kanad_api_url") or cfg.get("kanad_url") or "").rstrip("/")
        if not base:
            console.print("[red]No kanad API URL. Pass --api-url or set kanad_api_url in config.[/red]")
            raise SystemExit(1)
        try:
            r = httpx.get(f"{base}/api/compute/platform-key", timeout=10)
            r.raise_for_status()
            pubkey = r.json()["public_key"].strip()
        except Exception as e:
            console.print(f"[red]Could not fetch platform key from {base}: {e}[/red]")
            raise SystemExit(1)

    # Locked-down entry: restrict everything, then re-allow ONLY a local
    # port-forward to the compute API. asyncssh needs nothing more.
    opts = f'restrict,permitopen="127.0.0.1:{port}"'
    entry = f"{opts} {pubkey}".strip()
    keybody = " ".join(pubkey.split()[:2])  # 'keytype base64' for idempotency check

    ak = os.path.expanduser("~/.ssh/authorized_keys")
    os.makedirs(os.path.dirname(ak), mode=0o700, exist_ok=True)
    existing = ""
    if os.path.exists(ak):
        with open(ak) as f:
            existing = f.read()
    if keybody and keybody in existing:
        console.print("[yellow]Kanad platform key is already authorized.[/yellow]")
        return
    with open(ak, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(entry + "\n")
    os.chmod(ak, 0o600)
    console.print(Panel(
        f"[bold green]Authorized kanad-app to connect over SSH.[/bold green]\n\n"
        f"[dim]Locked to a local port-forward → 127.0.0.1:{port} only (no shell).[/dim]\n"
        f"[dim]Run [bold]kanad-compute pair[/bold] to see your pairing details.[/dim]",
        title="[bold green]Authorized[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()

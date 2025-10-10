from datetime import datetime
from rich.console import Console
console = Console()

def log(msg: str, style="cyan"):
    console.print(f"[{datetime.now():%H:%M:%S}] [{style}]{msg}[/]")

class TimeoutException(Exception):
    """Raised when a score-function exceeds its time budget."""
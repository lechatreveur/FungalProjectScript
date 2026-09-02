from pathlib import Path
from .errors import SecurityError

def resolve_under_root(base_root: Path, *parts: str) -> Path:
    """Safely resolve path under base_root and prevent path traversal."""
    resolved_base = base_root.resolve()
    target = resolved_base.joinpath(*parts).resolve()
    try:
        target.relative_to(resolved_base)
    except ValueError:
        raise SecurityError(f"Path traversal detected: {target} is outside {resolved_base}")
    return target

from pathlib import Path
import subprocess
import re
import logging
from typing import Sequence, Any

logger = logging.getLogger(__name__)

# Allowed path component regex: alphanumeric, underscores, hyphens, dots
COMPONENT_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

def validate_path_component(component: str, name: str = "Path component") -> str:
    """Validate that a path component is safe and contains no directory separators or traversal sequences."""
    if not component or component in {".", ".."}:
        raise ValueError(f"{name} cannot be empty or relative path navigation")
    if "/" in component or "\\" in component:
        raise ValueError(f"{name} cannot contain path separators")
    if not COMPONENT_REGEX.match(component):
        raise ValueError(f"{name} contains invalid characters: '{component}'")
    return component

def resolve_under_root(root: Path, *parts: str) -> Path:
    """Safely resolve parts under root, ensuring the final path does not escape root."""
    resolved_root = root.resolve()
    current = resolved_root
    for part in parts:
        valid_part = validate_path_component(part)
        current = current / valid_part
        
    final_path = current.resolve()
    # Ensure final_path is inside resolved_root
    if final_path != resolved_root and resolved_root not in final_path.parents:
        raise ValueError(f"Path traversal detected: '{final_path}' escapes root '{resolved_root}'")
    return final_path

def safe_subprocess_run(cmd: Sequence[str | Path], check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run a subprocess securely using argument arrays without shell=True."""
    if kwargs.get("shell", False):
        logger.warning("shell=True requested in safe_subprocess_run; forcing shell=False for security.")
        kwargs["shell"] = False
        
    cmd_str_args = [str(arg) for arg in cmd]
    logger.info("Executing command: %s", " ".join(cmd_str_args))
    return subprocess.run(cmd_str_args, check=check, **kwargs)

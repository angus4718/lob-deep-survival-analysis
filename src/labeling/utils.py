

def ms_to_suffix(ms: int) -> str:
    """Convert milliseconds to readable time suffix."""
    if ms >= 60000:
        return f"{ms // 1000}s"
    elif ms >= 1000:
        return f"{ms / 1000:.1f}s".rstrip("0").rstrip(".")
    else:
        return f"{ms}ms"
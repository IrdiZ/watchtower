"""URL normalisation: the same address must always map to the same portal id."""
from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, parts.query, ""))  # drop #fragment


def portal_id_for(url: str) -> str:
    return "p_" + hashlib.sha1(normalize_url(url).encode()).hexdigest()[:12]


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()

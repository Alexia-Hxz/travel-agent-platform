"""Keep the photo CDN reachable when the local DNS answer for it is a black hole.

Measured on this network: ``images.pexels.com`` resolved to a single address
(103.73.220.77) that accepted no TCP connection on :443, while ``api.pexels.com``
resolved through Cloudflare and answered in 0.2 s. The photo **search** therefore
succeeded and returned real URLs, but every candidate then failed its reachability
probe, was rejected, and each recommendation card fell back to a bundled placeholder
-- a plan whose pictures had apparently been deleted, with no error anywhere.

The host itself is healthy; only the address was wrong. Two addresses for the same
host were verified to serve it with a valid certificate (104.18.66.220, 172.64.147.160),
and pinning one restored the whole picture pipeline.

This module only interferes when it has to. For a configured host it first asks the
normal resolver whether *that* address actually connects; if it does, nothing is
touched and a repaired network needs no configuration change. Only when the resolver's
own answer is unreachable does it pin one of the configured fallback addresses.

Configuration lives in ``.env`` as a comma-separated list of ``host=address`` entries,
each address list separated by ``|`` or spaces::

    IMAGE_HOST_FALLBACKS=images.pexels.com=104.18.66.220|104.18.67.220|172.64.147.160

With no such entry the module does nothing at all.
"""

from __future__ import annotations

from pathlib import Path
import logging
import os
import socket
import threading
import time

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
ENV_VAR = "IMAGE_HOST_FALLBACKS"
# Short on purpose. Five addresses that all time out would otherwise delay the app's
# first picture lookup by their full sum; a real CDN edge answers in well under a second.
PROBE_TIMEOUT = 1.5
# A host left unpinned is re-examined occasionally, so a network that comes back (or an
# address that dies) is picked up without restarting the process. A pin is kept for the
# life of the process: a working edge address is stable, and re-probing costs a socket.
RETRY_SECONDS = 120.0

_original_getaddrinfo = socket.getaddrinfo
_lock = threading.RLock()
_installed = False
_pins: dict[str, str] = {}
_unpinned_until: dict[str, float] = {}


def _configured() -> dict[str, list[str]]:
    """Parse the fallback table. Returns {} when nothing is configured."""
    load_dotenv(ENV_PATH)
    raw = (os.getenv(ENV_VAR) or "").strip()
    if not raw:
        return {}
    table: dict[str, list[str]] = {}
    for entry in raw.split(","):
        host, separator, addresses = entry.partition("=")
        host = host.strip().lower()
        if not separator or not host:
            continue
        candidates = [part.strip() for part in addresses.replace("|", " ").split() if part.strip()]
        if candidates:
            table[host] = candidates
    return table


def _connects(address: str) -> bool:
    """True when ``address`` accepts a TCP connection on the HTTPS port."""
    try:
        with socket.create_connection((address, 443), timeout=PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def _resolved_addresses(host: str) -> list[str]:
    """Whatever the untouched resolver says for this host, in order."""
    try:
        infos = _original_getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return []
    seen: list[str] = []
    for info in infos:
        address = info[4][0]
        if address not in seen:
            seen.append(address)
    return seen


def _decide(host: str, candidates: list[str]) -> str | None:
    """The address to pin for ``host``, or None to leave resolution alone.

    The resolver's own answer wins whenever it works, so a healthy network is never
    redirected to a hard-coded address.
    """
    for address in _resolved_addresses(host):
        if _connects(address):
            return None
    for candidate in candidates:
        if _connects(candidate):
            return candidate
    return None


def _pin_for(host: str, candidates: list[str]) -> str | None:
    with _lock:
        pinned = _pins.get(host)
        if pinned is not None:
            return pinned
        if _unpinned_until.get(host, 0.0) > time.monotonic():
            return None
    # Probing outside the lock: two threads may resolve the same host at once, which
    # costs one duplicate probe and never blocks a lookup behind the network.
    chosen = _decide(host, candidates)
    with _lock:
        if chosen is not None:
            _pins[host] = chosen
            logger.info("pinned %s to %s: the local resolver's answer is unreachable", host, chosen)
        else:
            _unpinned_until[host] = time.monotonic() + RETRY_SECONDS
            logger.info("no reachable address for %s; leaving resolution to the system", host)
        return chosen


def install() -> None:
    """Redirect configured hosts to a reachable address. Safe to call more than once."""
    global _installed
    with _lock:
        if _installed:
            return
        table = _configured()
        if not table:
            return
        _installed = True

    def getaddrinfo(host, port, *args, **kwargs):
        candidates = table.get(str(host).lower())
        if candidates:
            pinned = _pin_for(str(host).lower(), candidates)
            if pinned is not None:
                host = pinned
        return _original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo
    # Decide once at startup rather than on the first picture lookup: that lookup runs
    # inside a bounded DNS budget, and spending it on probing would read as a failure.
    for host, candidates in table.items():
        _pin_for(host, candidates)


def reset() -> None:
    """Forget every decision and restore the original resolver. Intended for tests."""
    global _installed
    with _lock:
        _pins.clear()
        _unpinned_until.clear()
        if _installed:
            socket.getaddrinfo = _original_getaddrinfo
        _installed = False

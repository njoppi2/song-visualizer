"""Small client helper that never retries an analysis POST."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlsplit

from .schema import health_payload

DEFAULT_DEADLINE_SECONDS = 120.0


class ColdStartTimeout(TimeoutError):
    """The authenticated health route did not become reachable by the deadline."""


class ResultRedirectError(RuntimeError):
    """A long-running Modal result redirect cannot be followed safely."""


def follow_result_redirect(
    *,
    original_url: str,
    location: str | None,
    headers: Mapping[str, str],
    request: Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    """Fetch a Modal HTTP-303 result URL without resubmitting the analysis POST.

    Modal Web Functions can hand a request that outlives the HTTP deadline to a
    result URL.  The caller must capture the original response's ``Location``
    header and pass it here.  This function deliberately makes exactly one
    **GET** to that URL; it never repeats the original POST.  Proxy credentials
    are sent only when the result URL has the same scheme and authority as the
    original private endpoint.
    """
    if not location:
        raise ResultRedirectError("HTTP 303 response did not include a Location header")
    resolved_url = urljoin(original_url, location)
    original = urlsplit(original_url)
    resolved = urlsplit(resolved_url)
    if (
        original.scheme not in {"http", "https"}
        or resolved.scheme != original.scheme
        or resolved.netloc != original.netloc
        or resolved.username is not None
        or resolved.password is not None
    ):
        raise ResultRedirectError("refusing to send private credentials to a foreign result URL")

    status, body = request("GET", resolved_url, headers, None)
    if status != 200:
        raise ResultRedirectError(f"result GET failed with HTTP {status}; POST was not retried")
    return body


def invoke_once(
    *,
    base_url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    request: Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, Mapping[str, Any]]],
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Wait through cold start on health, then send precisely one analysis POST.

    Proxy-token headers are caller-owned and are neither persisted nor logged.
    Only 503 from health is retryable.  A POST is never retried because its
    execution state is ambiguous after transport failure.
    """
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    deadline = monotonic() + deadline_seconds
    delay = 1.0
    health_url = base_url.rstrip("/") + "/health"
    while True:
        status, body = request("GET", health_url, headers, None)
        if status == 200 and dict(body) == health_payload():
            break
        if status != 503 or monotonic() >= deadline:
            raise ColdStartTimeout("private health route did not become ready before deadline")
        sleep(min(delay, max(0.0, deadline - monotonic())))
        delay = min(delay * 2, 8.0)
    encoded = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
    post_headers = {**headers, "Content-Type": "application/json"}
    status, body = request("POST", base_url.rstrip("/") + "/analyze", post_headers, encoded)
    if status != 200:
        raise RuntimeError(f"analysis request failed with HTTP {status}; not retrying POST")
    return body

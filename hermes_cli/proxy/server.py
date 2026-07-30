"""HTTP server that forwards OpenAI-compatible requests to a configured upstream.

Listens on ``http://<host>:<port>/v1/<path>`` and forwards each request to
``<upstream-base-url>/<path>`` with the client's ``Authorization`` header
replaced by a freshly-resolved bearer from the configured adapter. The
response bytes are forwarded unmodified. Non-streaming responses are buffered so
upstream body failures can still return a precise proxy error; streaming/SSE
responses remain streamed.

The server does not log or transform request/response bodies. It is a
credential-attaching forwarder with bounded lifecycle telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import time
import uuid
from typing import Optional

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

# Headers we strip when forwarding to the upstream. ``host``/``content-length``
# are recomputed by aiohttp; ``authorization`` is replaced with our bearer.
# Everything else (content-type, accept, user-agent, x-* headers) passes through.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "authorization",  # we replace this one
    }
)

DEFAULT_PORT = 8645
DEFAULT_HOST = "127.0.0.1"
# Body cap for forwarded requests. Chat-completion payloads with long agent
# conversations can be large; mirror api_server's MAX_REQUEST_BYTES (10 MB).
# client_max_size bounds every read path, including chunked bodies.
MAX_REQUEST_BYTES = 10_000_000
MAX_BUFFERED_RESPONSE_BYTES = 32_000_000


_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _request_id(value: Optional[str]) -> str:
    candidate = str(value or "").strip()
    return candidate if _REQUEST_ID.fullmatch(candidate) else uuid.uuid4().hex


def _observed_request_id(value: Optional[str]) -> str:
    candidate = str(value or "").strip()
    return candidate if _REQUEST_ID.fullmatch(candidate) else "-"


def _json_error(
    status: int,
    message: str,
    code: str = "proxy_error",
    *,
    request_id: Optional[str] = None,
) -> "web.Response":
    """Return an OpenAI-style error JSON response."""
    body = {"error": {"message": message, "type": code, "code": code}}
    headers = {"X-Hermes-Proxy-Request-ID": request_id} if request_id else None
    return web.json_response(body, status=status, headers=headers)


def _filter_request_headers(headers: "aiohttp.typedefs.LooseHeaders") -> dict:
    """Strip hop-by-hop + auth headers from the inbound request."""
    out = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        out[key] = value
    return out


def _filter_response_headers(headers) -> dict:
    """Strip hop-by-hop headers from the upstream response."""
    out = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        # aiohttp recomputes Content-Encoding/Content-Length on stream — let it.
        if key.lower() in {"content-encoding", "content-length"}:
            continue
        out[key] = value
    return out


def _streaming_requested(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def create_app(
    adapter: UpstreamAdapter,
    *,
    upstream_sock_connect_seconds: float = 15,
    upstream_sock_read_seconds: float = 300,
) -> "web.Application":
    """Build the aiohttp application bound to a specific upstream adapter."""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for `hermes proxy`. Run `hermes setup` to install it."
        )

    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    # AppKey ensures forward-compat with future aiohttp versions that strip
    # bare-string keys.
    _adapter_key = web.AppKey("adapter", UpstreamAdapter)
    app[_adapter_key] = adapter

    async def handle_health(request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "upstream": adapter.display_name,
                "authenticated": adapter.is_authenticated(),
            }
        )

    async def handle_proxy(request: "web.Request") -> "web.StreamResponse":
        request_id = _request_id(request.headers.get("X-Request-ID"))
        started = time.monotonic()

        def duration_ms() -> int:
            return int((time.monotonic() - started) * 1000)

        # Extract the path *after* /v1
        rel_path = request.match_info.get("tail", "")
        rel_path = "/" + rel_path.lstrip("/")
        logger.info(
            "event=proxy_request_started request_id=%s provider=%s method=%s path=%s "
            "connect_timeout_seconds=%s read_timeout_seconds=%s",
            request_id,
            adapter.name,
            request.method,
            rel_path,
            upstream_sock_connect_seconds,
            upstream_sock_read_seconds,
        )

        if rel_path not in adapter.allowed_paths:
            allowed = ", ".join(sorted(adapter.allowed_paths))
            logger.info(
                "event=proxy_request_failed request_id=%s provider=%s stage=admission "
                "code=path_not_allowed duration_ms=%d",
                request_id,
                adapter.name,
                duration_ms(),
            )
            return _json_error(
                404,
                f"Path /v1{rel_path} is not forwarded by this proxy. "
                f"Allowed: {allowed}",
                code="path_not_allowed",
                request_id=request_id,
            )

        try:
            cred = adapter.get_credential()
        except Exception as exc:
            logger.warning("proxy: credential resolution failed: %s", exc)
            logger.info(
                "event=proxy_request_failed request_id=%s provider=%s stage=credential "
                "code=upstream_auth_failed duration_ms=%d",
                request_id,
                adapter.name,
                duration_ms(),
            )
            return _json_error(
                401,
                str(exc),
                code="upstream_auth_failed",
                request_id=request_id,
            )

        # Forward body verbatim. Read into memory once — request bodies for
        # chat/completions/embeddings are small (<1MB typically). If we ever
        # need to forward large multipart uploads we'll switch to streaming
        # the request body too.
        body = await request.read()

        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=upstream_sock_connect_seconds,
            sock_read=upstream_sock_read_seconds,
        )

        async def _send_upstream(active_cred: UpstreamCredential):
            upstream_url = f"{active_cred.base_url.rstrip('/')}{rel_path}"
            # Preserve query string verbatim.
            if request.query_string:
                upstream_url = f"{upstream_url}?{request.query_string}"

            fwd_headers = _filter_request_headers(request.headers)
            fwd_headers["Authorization"] = f"{active_cred.token_type} {active_cred.bearer}"
            fwd_headers["X-Request-ID"] = request_id

            logger.debug(
                "proxy: forwarding %s %s -> %s (body=%d bytes)",
                request.method, rel_path, upstream_url, len(body),
            )

            try:
                session = aiohttp.ClientSession(timeout=timeout)
            except Exception as exc:  # pragma: no cover - aiohttp setup issue
                raise RuntimeError(f"proxy session init failed: {exc}") from exc

            try:
                upstream_resp = await session.request(
                    request.method,
                    upstream_url,
                    data=body if body else None,
                    headers=fwd_headers,
                    allow_redirects=False,
                )
            except asyncio.CancelledError:
                await session.close()
                raise
            except Exception:
                await session.close()
                raise
            return session, upstream_resp

        async def _open_upstream(active_cred: UpstreamCredential):
            try:
                return await _send_upstream(active_cred)
            except RuntimeError as exc:
                logger.info(
                    "event=proxy_request_failed request_id=%s provider=%s "
                    "stage=proxy_session code=proxy_session_error duration_ms=%d",
                    request_id,
                    adapter.name,
                    duration_ms(),
                )
                return _json_error(
                    500,
                    str(exc),
                    code="proxy_session_error",
                    request_id=request_id,
                ), None
            except asyncio.TimeoutError:
                logger.info(
                    "event=proxy_request_failed request_id=%s provider=%s "
                    "stage=upstream_wait code=upstream_timeout duration_ms=%d",
                    request_id,
                    adapter.name,
                    duration_ms(),
                )
                return (
                    _json_error(
                        504,
                        "upstream request timed out",
                        code="upstream_timeout",
                        request_id=request_id,
                    ),
                    None,
                )
            except aiohttp.ClientError as exc:
                logger.warning("proxy: upstream connection failed: %s", exc)
                logger.info(
                    "event=proxy_request_failed request_id=%s provider=%s "
                    "stage=upstream_connect code=upstream_unreachable duration_ms=%d",
                    request_id,
                    adapter.name,
                    duration_ms(),
                )
                return (
                    _json_error(
                        502,
                        f"upstream connection failed: {exc}",
                        code="upstream_unreachable",
                        request_id=request_id,
                    ),
                    None,
                )

        try:
            session_or_response, upstream_resp = await _open_upstream(cred)
        except asyncio.CancelledError:
            logger.info(
                "event=proxy_request_failed request_id=%s provider=%s "
                "stage=upstream_wait code=client_cancelled duration_ms=%d",
                request_id,
                adapter.name,
                duration_ms(),
            )
            raise
        if upstream_resp is None:
            return session_or_response
        session = session_or_response
        logger.info(
            "event=proxy_upstream_headers request_id=%s provider=%s status=%d "
            "upstream_request_id=%s duration_ms=%d",
            request_id,
            adapter.name,
            upstream_resp.status,
            _observed_request_id(upstream_resp.headers.get("x-request-id")),
            duration_ms(),
        )

        if upstream_resp.status in {401, 429}:
            try:
                retry_cred = adapter.get_retry_credential(
                    failed_credential=cred,
                    status_code=upstream_resp.status,
                )
            except Exception as exc:
                logger.warning("proxy: retry credential resolution failed: %s", exc)
                retry_cred = None

            if retry_cred is not None:
                upstream_resp.release()
                await session.close()
                session_or_response, upstream_resp = await _open_upstream(retry_cred)
                if upstream_resp is None:
                    return session_or_response
                session = session_or_response
                logger.info(
                    "event=proxy_upstream_headers request_id=%s provider=%s status=%d "
                    "upstream_request_id=%s credential_attempt=2 duration_ms=%d",
                    request_id,
                    adapter.name,
                    upstream_resp.status,
                    _observed_request_id(upstream_resp.headers.get("x-request-id")),
                    duration_ms(),
                )

        response_headers = {
            **_filter_response_headers(upstream_resp.headers),
            "X-Hermes-Proxy-Request-ID": request_id,
        }

        if not _streaming_requested(body):
            buffered = bytearray()
            try:
                async for chunk in upstream_resp.content.iter_any():
                    if not chunk:
                        continue
                    buffered.extend(chunk)
                    if len(buffered) > MAX_BUFFERED_RESPONSE_BYTES:
                        logger.info(
                            "event=proxy_request_failed request_id=%s provider=%s "
                            "stage=upstream_body code=response_too_large status=%d "
                            "duration_ms=%d bytes_read=%d",
                            request_id,
                            adapter.name,
                            upstream_resp.status,
                            duration_ms(),
                            len(buffered),
                        )
                        return _json_error(
                            502,
                            "upstream response exceeded proxy buffer limit",
                            code="response_too_large",
                            request_id=request_id,
                        )
            except asyncio.TimeoutError:
                logger.info(
                    "event=proxy_request_failed request_id=%s provider=%s "
                    "stage=upstream_body code=upstream_timeout status=%d "
                    "duration_ms=%d bytes_read=%d",
                    request_id,
                    adapter.name,
                    upstream_resp.status,
                    duration_ms(),
                    len(buffered),
                )
                return _json_error(
                    504,
                    "upstream response body timed out",
                    code="upstream_timeout",
                    request_id=request_id,
                )
            except asyncio.CancelledError:
                logger.info(
                    "event=proxy_request_failed request_id=%s provider=%s "
                    "stage=upstream_body code=client_cancelled status=%d "
                    "duration_ms=%d bytes_read=%d",
                    request_id,
                    adapter.name,
                    upstream_resp.status,
                    duration_ms(),
                    len(buffered),
                )
                raise
            except aiohttp.ClientError as exc:
                logger.warning("proxy: upstream body read failed: %s", exc)
                logger.info(
                    "event=proxy_request_failed request_id=%s provider=%s "
                    "stage=upstream_body code=upstream_body_failed status=%d "
                    "duration_ms=%d bytes_read=%d",
                    request_id,
                    adapter.name,
                    upstream_resp.status,
                    duration_ms(),
                    len(buffered),
                )
                return _json_error(
                    502,
                    "upstream response body failed",
                    code="upstream_body_failed",
                    request_id=request_id,
                )
            finally:
                upstream_resp.release()
                await session.close()

            logger.info(
                "event=proxy_request_completed request_id=%s provider=%s status=%d "
                "duration_ms=%d bytes_out=%d",
                request_id,
                adapter.name,
                upstream_resp.status,
                duration_ms(),
                len(buffered),
            )
            return web.Response(
                body=bytes(buffered),
                status=upstream_resp.status,
                headers=response_headers,
            )

        # Streaming responses forward headers and chunks as they arrive.
        resp = web.StreamResponse(
            status=upstream_resp.status,
            headers=response_headers,
        )
        bytes_out = 0
        try:
            await resp.prepare(request)
            async for chunk in upstream_resp.content.iter_any():
                if chunk:
                    bytes_out += len(chunk)
                    await resp.write(chunk)
            await resp.write_eof()
        except asyncio.CancelledError:
            logger.info(
                "event=proxy_request_failed request_id=%s provider=%s "
                "stage=response_stream code=client_cancelled status=%d "
                "duration_ms=%d bytes_out=%d",
                request_id,
                adapter.name,
                upstream_resp.status,
                duration_ms(),
                bytes_out,
            )
            raise
        except aiohttp.ClientError as exc:
            logger.warning("proxy: streaming interrupted: %s", exc)
            logger.info(
                "event=proxy_request_failed request_id=%s provider=%s "
                "stage=response_stream code=stream_interrupted status=%d "
                "duration_ms=%d bytes_out=%d",
                request_id,
                adapter.name,
                upstream_resp.status,
                duration_ms(),
                bytes_out,
            )
            return resp
        except ConnectionError:
            logger.info(
                "event=proxy_request_failed request_id=%s provider=%s "
                "stage=response_stream code=client_disconnected status=%d "
                "duration_ms=%d bytes_out=%d",
                request_id,
                adapter.name,
                upstream_resp.status,
                duration_ms(),
                bytes_out,
            )
            return resp
        finally:
            upstream_resp.release()
            await session.close()

        logger.info(
            "event=proxy_request_completed request_id=%s provider=%s status=%d "
            "duration_ms=%d bytes_out=%d",
            request_id,
            adapter.name,
            upstream_resp.status,
            duration_ms(),
            bytes_out,
        )
        return resp

    # /health doesn't go through the upstream
    app.router.add_get("/health", handle_health)
    # Catch-all under /v1 — forwards if the path is allowed.
    app.router.add_route("*", "/v1/{tail:.*}", handle_proxy)

    return app


async def run_server(
    adapter: UpstreamAdapter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    shutdown_event: Optional[asyncio.Event] = None,
    upstream_sock_connect_seconds: float = 15,
    upstream_sock_read_seconds: float = 300,
) -> None:
    """Run the proxy in the current event loop until shutdown_event is set.

    If shutdown_event is None, runs until cancelled (Ctrl+C or SIGTERM).
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for `hermes proxy`. Run `hermes setup` to install it."
        )

    app = create_app(
        adapter,
        upstream_sock_connect_seconds=upstream_sock_connect_seconds,
        upstream_sock_read_seconds=upstream_sock_read_seconds,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    logger.info(
        "proxy: listening on http://%s:%d/v1 -> %s "
        "(connect_timeout_seconds=%s read_timeout_seconds=%s)",
        host,
        port,
        adapter.display_name,
        upstream_sock_connect_seconds,
        upstream_sock_read_seconds,
    )

    stop_event = shutdown_event or asyncio.Event()

    # Wire signal handlers when we own the loop's lifetime.
    if shutdown_event is None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)  # windows-footgun: ok
            except NotImplementedError:
                # Windows / restricted environments — Ctrl+C will still
                # raise KeyboardInterrupt and unwind us.
                pass

    try:
        await stop_event.wait()
    finally:
        logger.info("proxy: shutting down")
        await runner.cleanup()


__all__ = [
    "create_app",
    "run_server",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "AIOHTTP_AVAILABLE",
]

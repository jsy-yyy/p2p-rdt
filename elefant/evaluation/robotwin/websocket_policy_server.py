from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback

import websockets
import websockets.asyncio.server as websockets_server
import websockets.frames

from .msgpack_numpy import Packer, unpackb

LOGGER = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve a stateful policy over websocket + msgpack."""

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets_server.ServerConnection) -> None:
        LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                payload = unpackb(await websocket.recv())
                infer_start = time.monotonic()
                response = self._policy.infer(payload)
                infer_time = time.monotonic() - infer_start

                response["server_timing"] = {"infer_ms": infer_time * 1000.0}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: websockets_server.ServerConnection,
    request: websockets_server.Request,
) -> websockets_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None

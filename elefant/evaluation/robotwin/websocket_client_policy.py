from __future__ import annotations

import logging
import time
from typing import Any

import websockets.sync.client

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Simple websocket client for remote RoboTwin policy inference."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for server at %s", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError, TimeoutError, Exception) as exc:
                logging.info("Still waiting for server: %s", exc)
                time.sleep(5)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(
        self,
        prompt: str | None = None,
        text_emb: Any | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"reset": True}
        if prompt is not None:
            payload["prompt"] = prompt
        if text_emb is not None:
            payload["text_emb"] = text_emb
        return self.infer(payload)

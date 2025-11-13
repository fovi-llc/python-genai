"""Python interface for Chrome's Prompt API."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from IPython.display import display

from ._widget import _STREAM_END, PromptWidget


class PromptError(RuntimeError):
    """Raised when the underlying Prompt API reports a failure."""


@dataclass
class PromptStream(AsyncIterator[Any]):
    """Async iterator for streaming prompt responses."""

    queue: "asyncio.Queue[Any]"
    future: "asyncio.Future[Any]"
    request_id: str
    widget: PromptWidget
    _closed: bool = False

    def __aiter__(self) -> "PromptStream":  # pragma: no cover - trivial delegation
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        item = await self.queue.get()
        if item is _STREAM_END:
            self._closed = True
            raise StopAsyncIteration
        if isinstance(item, Exception):
            self._closed = True
            raise PromptError(str(item))
        return item

    async def result(self) -> Any:
        """Wait for the stream to finish and return the terminal result."""

        return await self.future

    def close(self) -> None:
        """Terminate the stream and release widget resources."""

        if self._closed:
            return
        self._closed = True
        self.widget.close_stream(self.request_id)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


class PromptSession:
    """Wraps a ``window.ai`` prompt session."""

    def __init__(self, widget: PromptWidget, session_id: str) -> None:
        self._widget = widget
        self._session_id = session_id

    async def prompt(self, text: str, options: Optional[Dict[str, Any]] = None) -> Any:
        """Submit ``text`` to the session and return the response."""

        params = {"sessionId": self._session_id, "prompt": text}
        if options:
            params["options"] = options
        return await self._widget.call("session.prompt", params)

    async def prompt_stream(self, text: str, options: Optional[Dict[str, Any]] = None) -> PromptStream:
        """Stream the response for ``text``."""

        params = {"sessionId": self._session_id, "prompt": text}
        if options:
            params["options"] = options
        return await self._widget.stream("session.promptStreaming", params)

    async def can_prompt(self, options: Optional[Dict[str, Any]] = None) -> Any:
        """Check whether the session can accept a prompt with the given options."""

        params = {"sessionId": self._session_id}
        if options:
            params["options"] = options
        return await self._widget.call("session.canPrompt", params)

    async def destroy(self) -> None:
        """Dispose the underlying session."""

        await self._widget.call("session.destroy", {"sessionId": self._session_id})

    async def clone(self) -> "PromptSession":
        """Clone the session, returning a new :class:`PromptSession`."""

        new_session_id = uuid.uuid4().hex
        await self._widget.call(
            "session.clone",
            {"sessionId": self._session_id, "newSessionId": new_session_id},
        )
        return PromptSession(self._widget, new_session_id)

    async def __aenter__(self) -> "PromptSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.destroy()


class PromptAPI:
    """High-level entry point for the Prompt API."""

    def __init__(self, *, widget: Optional[PromptWidget] = None, auto_display: bool = True) -> None:
        self._widget = widget or PromptWidget()
        if auto_display:
            display(self._widget)

    @property
    def widget(self) -> PromptWidget:
        return self._widget

    async def get_capabilities(self) -> Any:
        """Return ``window.ai`` capabilities."""

        return await self._widget.call("getCapabilities")

    async def create_text_session(self, options: Optional[Dict[str, Any]] = None) -> PromptSession:
        """Create a text prompt session."""

        session_id = uuid.uuid4().hex
        params = {"sessionId": session_id}
        if options:
            params["options"] = options
        await self._widget.call("createTextSession", params)
        return PromptSession(self._widget, session_id)


__all__ = ["PromptAPI", "PromptSession", "PromptStream", "PromptError"]

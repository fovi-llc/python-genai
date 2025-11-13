"""AnyWidget bridge for Chrome Prompt API."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import anywidget
from traitlets import Dict as TraitletDict


@dataclass
class _StreamState:
    queue: "asyncio.Queue[Any]"
    result: "asyncio.Future[Any]"


_STREAM_END = object()


class PromptWidget(anywidget.AnyWidget):
    """Widget providing a thin RPC layer over Chrome's ``window.ai`` Prompt API."""

    request = TraitletDict(allow_none=True, default_value=None).tag(sync=True)
    response = TraitletDict(allow_none=True, default_value=None).tag(sync=True)
    notification = TraitletDict(allow_none=True, default_value=None).tag(sync=True)

    _esm = r"""
    export default function PromptWidget({model}) {
      const sessions = new Map();

      function addStamp(payload) {
        return {
          ...payload,
          __stamp: Math.random().toString(36).slice(2)
        };
      }

      function stripStamp(payload) {
        if (!payload || typeof payload !== 'object') {
          return payload;
        }
        const { __stamp, ...rest } = payload;
        return rest;
      }

      function setResponse(payload) {
        model.set('response', addStamp(payload));
        model.save_changes();
      }

      function setNotification(payload) {
        model.set('notification', addStamp(payload));
        model.save_changes();
      }

      async function handleRequest(raw) {
        if (!raw) {
          return;
        }
        const payload = stripStamp(raw);
        const { id, method, params = {} } = payload;

        const ensureSession = () => {
          const session = sessions.get(params.sessionId);
          if (!session) {
            throw new Error('Prompt session not found');
          }
          return session;
        };

        const wrapError = (error) => ({
          id,
          error: {
            name: error?.name ?? 'Error',
            message: error?.message ?? String(error),
          },
        });

        try {
          switch (method) {
            case 'getCapabilities': {
              const result = await window?.ai?.getCapabilities?.();
              setResponse({ id, result });
              break;
            }
            case 'createTextSession': {
              if (!window?.ai?.createTextSession) {
                throw new Error('window.ai.createTextSession is unavailable');
              }
              const options = params.options ?? {};
              const session = await window.ai.createTextSession(options);
              sessions.set(params.sessionId, session);
              setResponse({ id, result: true });
              break;
            }
            case 'session.prompt': {
              const session = ensureSession();
              const result = await session.prompt(params.prompt, params.options ?? {});
              setResponse({ id, result });
              break;
            }
            case 'session.promptStreaming': {
              const session = ensureSession();
              const stream = await session.promptStreaming(params.prompt, params.options ?? {});
              const reader = stream?.getReader?.();
              if (!reader) {
                throw new Error('Prompt stream reader unavailable');
              }
              (async () => {
                try {
                  while (true) {
                    const { value, done } = await reader.read();
                    if (done) {
                      break;
                    }
                    setNotification({ id, event: 'chunk', value });
                  }
                  setNotification({ id, event: 'end', result: true });
                } catch (streamError) {
                  setNotification({ id, event: 'error', error: {
                    name: streamError?.name ?? 'Error',
                    message: streamError?.message ?? String(streamError),
                  }});
                }
              })();
              setResponse({ id, result: true });
              break;
            }
            case 'session.canPrompt': {
              const session = ensureSession();
              const result = await session.canPrompt(params.options ?? {});
              setResponse({ id, result });
              break;
            }
            case 'session.clone': {
              const session = ensureSession();
              const cloned = await session.clone();
              sessions.set(params.newSessionId, cloned);
              setResponse({ id, result: true });
              break;
            }
            case 'session.destroy': {
              const session = ensureSession();
              await session.destroy();
              sessions.delete(params.sessionId);
              setResponse({ id, result: true });
              break;
            }
            default:
              throw new Error(`Unknown method: ${method}`);
          }
        } catch (error) {
          setResponse(wrapError(error));
        }
      }

      model.on('change:request', () => {
        const raw = model.get('request');
        if (raw) {
          handleRequest(raw);
        }
      });
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._futures: Dict[str, "asyncio.Future[Any]"] = {}
        self._streams: Dict[str, _StreamState] = {}
        self.observe(self._handle_response, names="response")
        self.observe(self._handle_notification, names="notification")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        return loop

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        loop = self._ensure_loop()
        future: "asyncio.Future[Any]" = loop.create_future()
        request_id = uuid.uuid4().hex
        self._futures[request_id] = future
        payload = {"id": request_id, "method": method, "params": params or {}}
        self._emit_request(payload)
        return await future

    async def stream(self, method: str, params: Optional[Dict[str, Any]] = None) -> "PromptStream":
        loop = self._ensure_loop()
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        result: "asyncio.Future[Any]" = loop.create_future()
        request_id = uuid.uuid4().hex
        self._streams[request_id] = _StreamState(queue=queue, result=result)
        payload = {
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self._emit_request(payload)
        from .prompt import PromptStream  # local import to avoid circular

        return PromptStream(queue=queue, future=result, request_id=request_id, widget=self)

    def _handle_response(self, change: Dict[str, Any]) -> None:
        raw = change.get("new")
        payload = self._strip_stamp(raw)
        if not payload:
            return
        request_id = payload.get("id")
        if not request_id:
            return
        future = self._futures.pop(request_id, None)
        if not future or future.done():
            return
        error = payload.get("error")
        if error:
            future.set_exception(RuntimeError(error.get("message", "Prompt API error")))
        else:
            future.set_result(payload.get("result"))

    def _handle_notification(self, change: Dict[str, Any]) -> None:
        raw = change.get("new")
        payload = self._strip_stamp(raw)
        if not payload:
            return
        request_id = payload.get("id")
        if not request_id:
            return
        state = self._streams.get(request_id)
        if not state:
            return
        event = payload.get("event")
        if event == "chunk":
            state.queue.put_nowait(payload.get("value"))
        elif event == "end":
            if not state.result.done():
                state.result.set_result(payload.get("result"))
            state.queue.put_nowait(_STREAM_END)
            self._streams.pop(request_id, None)
        elif event == "error":
            message = "Prompt stream error"
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message", message)
            error = RuntimeError(message)
            if not state.result.done():
                state.result.set_exception(error)
            state.queue.put_nowait(error)
            state.queue.put_nowait(_STREAM_END)
            self._streams.pop(request_id, None)

    def close_stream(self, request_id: str) -> None:
        state = self._streams.pop(request_id, None)
        if not state:
            return
        if not state.result.done():
            state.result.set_exception(RuntimeError("Prompt stream closed"))
        state.queue.put_nowait(RuntimeError("Prompt stream closed"))
        state.queue.put_nowait(_STREAM_END)

    def _emit_request(self, payload: Dict[str, Any]) -> None:
        stamped = dict(payload)
        stamped["__stamp"] = uuid.uuid4().hex
        self.request = stamped

    @staticmethod
    def _strip_stamp(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        cleaned = dict(payload)
        cleaned.pop("__stamp", None)
        return cleaned


__all__ = ["PromptWidget", "_STREAM_END", "_StreamState"]

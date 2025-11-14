"""Language Model class implementation using AnyWidget for Jupyter integration."""

import asyncio
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import anywidget
import traitlets

from .models import (
    Availability,
    LanguageModelAppendOptions,
    LanguageModelCloneOptions,
    LanguageModelCreateOptions,
    LanguageModelMessage,
    LanguageModelParams,
    LanguageModelPromptOptions,
)


class LanguageModelWidget(anywidget.AnyWidget):
    """AnyWidget bridge to Chrome's Prompt API."""

    _esm = """
    function render({ model, el }) {
        // Check if Chrome Prompt API is available
        if (!window.ai || !window.ai.languageModel) {
            model.set('error', {
                type: 'NotSupportedError',
                message: 'Chrome Prompt API is not available in this browser'
            });
            return;
        }

        // Handle requests from Python
        model.on('change:request', async () => {
            const request = model.get('request');
            if (!request || !request.id) return;

            try {
                const result = await handleRequest(request);
                model.set('response', {
                    id: request.id,
                    result: result,
                    error: null
                });
            } catch (error) {
                model.set('response', {
                    id: request.id,
                    result: null,
                    error: {
                        type: error.name || 'Error',
                        message: error.message || String(error)
                    }
                });
            }
        });

        async function handleRequest(request) {
            const { method, params } = request;

            switch (method) {
                case 'create':
                    return await createSession(params);
                case 'availability':
                    return await window.ai.languageModel.availability(params.options || {});
                case 'params':
                    return await window.ai.languageModel.params();
                case 'prompt':
                    return await promptSession(params);
                case 'promptStreaming':
                    return await promptStreamingSession(params);
                case 'append':
                    return await appendSession(params);
                case 'measureInputUsage':
                    return await measureInputUsageSession(params);
                case 'clone':
                    return await cloneSession(params);
                case 'destroy':
                    return await destroySession(params);
                default:
                    throw new Error(`Unknown method: ${method}`);
            }
        }

        // Store sessions by ID
        const sessions = {};

        async function createSession(params) {
            const session = await window.ai.languageModel.create(params.options || {});
            const sessionId = params.sessionId;
            sessions[sessionId] = session;

            // Set up quota overflow listener
            session.addEventListener('quotaoverflow', () => {
                model.set('quota_overflow_event', {
                    sessionId: sessionId,
                    timestamp: Date.now()
                });
            });

            return {
                sessionId: sessionId,
                topK: session.topK,
                temperature: session.temperature,
                inputUsage: session.inputUsage,
                inputQuota: session.inputQuota
            };
        }

        async function promptSession(params) {
            const session = sessions[params.sessionId];
            if (!session) throw new Error('Session not found');

            const result = await session.prompt(params.input, params.options || {});
            return {
                result: result,
                inputUsage: session.inputUsage,
                inputQuota: session.inputQuota
            };
        }

        async function promptStreamingSession(params) {
            const session = sessions[params.sessionId];
            if (!session) throw new Error('Session not found');

            const stream = session.promptStreaming(params.input, params.options || {});
            const chunks = [];

            for await (const chunk of stream) {
                chunks.push(chunk);
                // Send intermediate chunks back
                model.set('stream_chunk', {
                    sessionId: params.sessionId,
                    requestId: params.requestId,
                    chunk: chunk,
                    timestamp: Date.now()
                });
            }

            return {
                result: chunks.join(''),
                inputUsage: session.inputUsage,
                inputQuota: session.inputQuota
            };
        }

        async function appendSession(params) {
            const session = sessions[params.sessionId];
            if (!session) throw new Error('Session not found');

            await session.append(params.input, params.options || {});
            return {
                inputUsage: session.inputUsage,
                inputQuota: session.inputQuota
            };
        }

        async function measureInputUsageSession(params) {
            const session = sessions[params.sessionId];
            if (!session) throw new Error('Session not found');

            const usage = await session.measureInputUsage(params.input, params.options || {});
            return { usage: usage };
        }

        async function cloneSession(params) {
            const session = sessions[params.sessionId];
            if (!session) throw new Error('Session not found');

            const cloned = await session.clone(params.options || {});
            const newSessionId = params.newSessionId;
            sessions[newSessionId] = cloned;

            // Set up quota overflow listener for cloned session
            cloned.addEventListener('quotaoverflow', () => {
                model.set('quota_overflow_event', {
                    sessionId: newSessionId,
                    timestamp: Date.now()
                });
            });

            return {
                sessionId: newSessionId,
                topK: cloned.topK,
                temperature: cloned.temperature,
                inputUsage: cloned.inputUsage,
                inputQuota: cloned.inputQuota
            };
        }

        async function destroySession(params) {
            const session = sessions[params.sessionId];
            if (!session) throw new Error('Session not found');

            session.destroy();
            delete sessions[params.sessionId];
            return { success: true };
        }
    }

    export default { render };
    """

    # Traitlets for communication
    request = traitlets.Dict({}).tag(sync=True)
    response = traitlets.Dict({}).tag(sync=True)
    error = traitlets.Dict({}).tag(sync=True)
    quota_overflow_event = traitlets.Dict({}).tag(sync=True)
    stream_chunk = traitlets.Dict({}).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._stream_chunks: Dict[str, List[str]] = {}
        self.observe(self._handle_response, names=["response"])
        self.observe(self._handle_error, names=["error"])
        self.observe(self._handle_stream_chunk, names=["stream_chunk"])

    def _handle_response(self, change):
        """Handle response from JavaScript."""
        response = change["new"]
        if not response or "id" not in response:
            return

        request_id = response["id"]
        if request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if response.get("error"):
                error = response["error"]
                future.set_exception(
                    Exception(
                        f"{error.get('type', 'Error')}: {error.get('message', 'Unknown error')}"
                    )
                )
            else:
                future.set_result(response.get("result"))

    def _handle_error(self, change):
        """Handle error from JavaScript."""
        error = change["new"]
        if error and error.get("message"):
            # Global error not tied to a specific request
            print(f"Error: {error.get('type', 'Error')}: {error.get('message')}")

    def _handle_stream_chunk(self, change):
        """Handle streaming chunk from JavaScript."""
        chunk_data = change["new"]
        if not chunk_data or "requestId" not in chunk_data:
            return

        request_id = chunk_data["requestId"]
        if request_id not in self._stream_chunks:
            self._stream_chunks[request_id] = []
        self._stream_chunks[request_id].append(chunk_data["chunk"])

    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a request to JavaScript and await response."""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self._pending_requests[request_id] = future

        self.request = {"id": request_id, "method": method, "params": params or {}}

        return await future


class LanguageModel:
    """Python interface to Chrome's Prompt API Language Model."""

    def __init__(self, widget: LanguageModelWidget, session_id: str, session_data: Dict[str, Any]):
        self._widget = widget
        self._session_id = session_id
        self._top_k = session_data.get("topK")
        self._temperature = session_data.get("temperature")
        self._input_usage = session_data.get("inputUsage", 0.0)
        self._input_quota = session_data.get("inputQuota", float("inf"))

    @classmethod
    async def create(
        cls, options: Optional[Union[LanguageModelCreateOptions, Dict[str, Any]]] = None
    ) -> "LanguageModel":
        """Create a new language model session."""
        widget = LanguageModelWidget()
        session_id = str(uuid.uuid4())

        if isinstance(options, LanguageModelCreateOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        params = {"sessionId": session_id, "options": options_dict}
        result = await widget.send_request("create", params)

        return cls(widget, session_id, result)

    @classmethod
    async def availability(
        cls, options: Optional[Union[LanguageModelCreateOptions, Dict[str, Any]]] = None
    ) -> Availability:
        """Check availability of the language model with given options."""
        widget = LanguageModelWidget()

        if isinstance(options, LanguageModelCreateOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        result = await widget.send_request("availability", {"options": options_dict})
        return Availability(result)

    @classmethod
    async def params(cls) -> Optional[LanguageModelParams]:
        """Get language model parameters."""
        widget = LanguageModelWidget()
        result = await widget.send_request("params")

        if result is None:
            return None

        return LanguageModelParams.from_dict(result)

    async def prompt(
        self,
        input: Union[str, List[LanguageModelMessage], List[Dict[str, Any]]],
        options: Optional[Union[LanguageModelPromptOptions, Dict[str, Any]]] = None,
    ) -> str:
        """Prompt the language model and return the result."""
        input_data = self._prepare_input(input)

        if isinstance(options, LanguageModelPromptOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        params = {"sessionId": self._session_id, "input": input_data, "options": options_dict}
        result = await self._widget.send_request("prompt", params)

        self._input_usage = result.get("inputUsage", self._input_usage)
        self._input_quota = result.get("inputQuota", self._input_quota)

        return result["result"]

    async def prompt_streaming(
        self,
        input: Union[str, List[LanguageModelMessage], List[Dict[str, Any]]],
        options: Optional[Union[LanguageModelPromptOptions, Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Prompt the language model and stream the result."""
        input_data = self._prepare_input(input)

        if isinstance(options, LanguageModelPromptOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        request_id = str(uuid.uuid4())
        self._widget._stream_chunks[request_id] = []

        params = {
            "sessionId": self._session_id,
            "requestId": request_id,
            "input": input_data,
            "options": options_dict,
        }

        # Start the request
        asyncio.create_task(self._widget.send_request("promptStreaming", params))

        # Yield chunks as they arrive
        chunk_index = 0
        while True:
            chunks = self._widget._stream_chunks.get(request_id, [])
            if chunk_index < len(chunks):
                yield chunks[chunk_index]
                chunk_index += 1
            else:
                # Check if the request is complete
                await asyncio.sleep(0.1)
                if request_id not in self._widget._pending_requests:
                    # Request completed, yield remaining chunks
                    chunks = self._widget._stream_chunks.get(request_id, [])
                    while chunk_index < len(chunks):
                        yield chunks[chunk_index]
                        chunk_index += 1
                    break

        # Clean up
        if request_id in self._widget._stream_chunks:
            del self._widget._stream_chunks[request_id]

    async def append(
        self,
        input: Union[str, List[LanguageModelMessage], List[Dict[str, Any]]],
        options: Optional[Union[LanguageModelAppendOptions, Dict[str, Any]]] = None,
    ) -> None:
        """Append messages to the session without prompting for a response."""
        input_data = self._prepare_input(input)

        if isinstance(options, LanguageModelAppendOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        params = {"sessionId": self._session_id, "input": input_data, "options": options_dict}
        result = await self._widget.send_request("append", params)

        self._input_usage = result.get("inputUsage", self._input_usage)
        self._input_quota = result.get("inputQuota", self._input_quota)

    async def measure_input_usage(
        self,
        input: Union[str, List[LanguageModelMessage], List[Dict[str, Any]]],
        options: Optional[Union[LanguageModelPromptOptions, Dict[str, Any]]] = None,
    ) -> float:
        """Measure how many tokens an input will consume."""
        input_data = self._prepare_input(input)

        if isinstance(options, LanguageModelPromptOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        params = {"sessionId": self._session_id, "input": input_data, "options": options_dict}
        result = await self._widget.send_request("measureInputUsage", params)

        return result["usage"]

    @property
    def input_usage(self) -> float:
        """Current input usage in tokens."""
        return self._input_usage

    @property
    def input_quota(self) -> float:
        """Maximum input quota in tokens."""
        return self._input_quota

    @property
    def top_k(self) -> Optional[int]:
        """Top-K sampling parameter."""
        return self._top_k

    @property
    def temperature(self) -> Optional[float]:
        """Temperature sampling parameter."""
        return self._temperature

    async def clone(
        self, options: Optional[Union[LanguageModelCloneOptions, Dict[str, Any]]] = None
    ) -> "LanguageModel":
        """Clone the current session."""
        new_session_id = str(uuid.uuid4())

        if isinstance(options, LanguageModelCloneOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        params = {
            "sessionId": self._session_id,
            "newSessionId": new_session_id,
            "options": options_dict,
        }
        result = await self._widget.send_request("clone", params)

        return LanguageModel(self._widget, new_session_id, result)

    async def destroy(self) -> None:
        """Destroy the session and free resources."""
        params = {"sessionId": self._session_id}
        await self._widget.send_request("destroy", params)

    def _prepare_input(
        self, input: Union[str, List[LanguageModelMessage], List[Dict[str, Any]]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Prepare input for sending to JavaScript."""
        if isinstance(input, str):
            return input
        elif isinstance(input, list):
            result = []
            for item in input:
                if isinstance(item, LanguageModelMessage):
                    result.append(item.to_dict())
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    raise TypeError(f"Invalid input item type: {type(item)}")
            return result
        else:
            raise TypeError(f"Invalid input type: {type(input)}")

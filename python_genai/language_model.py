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
        let statusTextArea = document.createElement("textarea", {readonly: true});
        statusTextArea.value = "Initializing...";
        // Check if Chrome Prompt API is available
        if (!('LanguageModel' in self)) {
            model.set('error', {
                type: 'NotSupportedError',
                message: 'Chrome Prompt API is not available in this browser'
            });
            model.save_changes();
            return;
        }
        statusTextArea.value = "Chrome Prompt API is available.";
        el.appendChild(statusTextArea);

        // Store sessions by ID
        const sessions = {};

        // Handle requests from Python
        model.on('change:request', () => {
            const request = model.get('request');
            if (!request || !request.id) return;

            handleRequest(request)
                .then(result => {
                    console.log('Result: ', result);
                    model.set('response', {
                        id: request.id,
                        result: result,
                        error: null
                    });
                    model.save_changes();
                })
                .catch(error => {
                    console.error('Error: ', error);
                    model.set('response', {
                        id: request.id,
                        result: null,
                        error: {
                            type: error.name || 'Error',
                            message: error.message || String(error)
                        }
                    });
                    model.save_changes();
                });
        });

        function handleRequest(request) {
            const { method, params } = request;

            switch (method) {
                case 'create':
                    console.log('Creating session with params:', params);
                    return createSession(params);
                case 'availability':
                    console.log('Checking availability with params:', params);
                    return self.LanguageModel.availability(params.options || {});
                case 'params':
                    //FIXME: This fails serialization for traitlets
                    return self.LanguageModel.params();
                case 'prompt':
                    return promptSession(params);
                case 'promptStreaming':
                    return promptStreamingSession(params);
                case 'append':
                    return appendSession(params);
                case 'measureInputUsage':
                    return measureInputUsageSession(params);
                case 'clone':
                    return cloneSession(params);
                case 'destroy':
                    return destroySession(params);
                default:
                    return Promise.reject(new Error(`Unknown method: ${method}`));
            }
        }

        function createSession(params) {
            return self.LanguageModel.create(params.options || {})
                .then(session => {
                    const sessionId = params.sessionId;
                    sessions[sessionId] = session;

                    // Set up quota overflow listener
                    session.addEventListener('quotaoverflow', () => {
                        model.set('quota_overflow_event', {
                            sessionId: sessionId,
                            timestamp: Date.now()
                        });
                        model.save_changes();
                    });

                    return {
                        sessionId: sessionId,
                        topK: session.topK,
                        temperature: session.temperature,
                        inputUsage: session.inputUsage,
                        inputQuota: session.inputQuota
                    };
                });
        }

        function promptSession(params) {
            const session = sessions[params.sessionId];
            if (!session) return Promise.reject(new Error('Session not found'));

            return session.prompt(params.input, params.options || {})
                .then(result => {
                    return {
                        result: result,
                        inputUsage: session.inputUsage,
                        inputQuota: session.inputQuota
                    };
                });
        }

        function promptStreamingSession(params) {
            const session = sessions[params.sessionId];
            if (!session) return Promise.reject(new Error('Session not found'));

            const stream = session.promptStreaming(params.input, params.options || {});
            const chunks = [];

            return consumeStream(stream, chunks, params.sessionId, params.requestId)
                .then(() => {
                    return {
                        result: chunks.join(''),
                        inputUsage: session.inputUsage,
                        inputQuota: session.inputQuota
                    };
                });
        }

        async function consumeStream(stream, chunks, sessionId, requestId) {
            for await (const chunk of stream) {
                chunks.push(chunk);
                // Send intermediate chunks back
                model.set('stream_chunk', {
                    sessionId: sessionId,
                    requestId: requestId,
                    chunk: chunk,
                    timestamp: Date.now()
                });
                model.save_changes();
            }
        }

        function appendSession(params) {
            const session = sessions[params.sessionId];
            if (!session) return Promise.reject(new Error('Session not found'));

            return session.append(params.input, params.options || {})
                .then(() => {
                    return {
                        inputUsage: session.inputUsage,
                        inputQuota: session.inputQuota
                    };
                });
        }

        function measureInputUsageSession(params) {
            const session = sessions[params.sessionId];
            if (!session) return Promise.reject(new Error('Session not found'));

            return session.measureInputUsage(params.input, params.options || {})
                .then(usage => {
                    return { usage: usage };
                });
        }

        function cloneSession(params) {
            const session = sessions[params.sessionId];
            if (!session) return Promise.reject(new Error('Session not found'));

            return session.clone(params.options || {})
                .then(cloned => {
                    const newSessionId = params.newSessionId;
                    sessions[newSessionId] = cloned;

                    // Set up quota overflow listener for cloned session
                    cloned.addEventListener('quotaoverflow', () => {
                        model.set('quota_overflow_event', {
                            sessionId: newSessionId,
                            timestamp: Date.now()
                        });
                        model.save_changes();
                    });

                    return {
                        sessionId: newSessionId,
                        topK: cloned.topK,
                        temperature: cloned.temperature,
                        inputUsage: cloned.inputUsage,
                        inputQuota: cloned.inputQuota
                    };
                });
        }

        function destroySession(params) {
            const session = sessions[params.sessionId];
            if (!session) return Promise.reject(new Error('Session not found'));

            session.destroy();
            delete sessions[params.sessionId];
            return Promise.resolve({ success: true });
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
        # self.observe(self._handle_response, names=["response"])
        # self.observe(self._handle_error, names=["error"])
        # self.observe(self._handle_stream_chunk, names=["stream_chunk"])

    @traitlets.observe("response")
    def _handle_response(self, change):
        """Handle response from JavaScript."""
        print('Handling response: ', change)
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
        else:
            print(f"Warning: Received response for unknown request ID: {request_id}")

    @traitlets.observe("error")
    def _handle_error(self, change):
        """Handle error from JavaScript."""
        print('Handling error: ', change)
        error = change["new"]
        if error and error.get("message"):
            # Global error not tied to a specific request
            print(f"Error: {error.get('type', 'Error')}: {error.get('message')}")

    @traitlets.observe("stream_chunk")
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
        print('Sending request ID: ', request_id)
        future = asyncio.Future()
        self._pending_requests[request_id] = future

        self.request = {"id": request_id, "method": method, "params": params or {}}
        print('Sent request: ', self.request)

        # Yield control to allow the event loop to process traitlet updates
        # This is necessary for the JavaScript response to be received and processed
        while not future.done():
            await asyncio.sleep(0)
        
        return await future


class LanguageModel:
    """Python interface to Chrome's Prompt API Language Model."""

    def __init__(self, widget: LanguageModelWidget | None = None):
        self.widget = widget or LanguageModelWidget()
        self._session_id = None
        self._top_k = None
        self._temperature = None
        self._input_usage = 0.0
        self._input_quota = float("inf")

    async def create(
        self, options: Optional[Union[LanguageModelCreateOptions, Dict[str, Any]]] = None
    ) -> str | None:
        """Create a new language model session."""
        session_id = str(uuid.uuid4())

        if isinstance(options, LanguageModelCreateOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        params = {"sessionId": session_id, "options": options_dict}
        result = await self.widget.send_request("create", params)
        print('Created session result: ', result)

        self._session_id = session_id
        self._top_k = result.get("topK")
        self._temperature = result.get("temperature")
        self._input_usage = result.get("inputUsage", 0.0)
        self._input_quota = result.get("inputQuota", float("inf"))

        return self.session_id

    async def availability(
        self, options: Optional[Union[LanguageModelCreateOptions, Dict[str, Any]]] = None
    ) -> Availability:
        """Check availability of the language model with given options."""
        if isinstance(options, LanguageModelCreateOptions):
            options_dict = options.to_dict()
        elif options is None:
            options_dict = {}
        else:
            options_dict = options

        result = await self.widget.send_request("availability", {"options": options_dict})
        return Availability(result)

    async def params(self) -> Optional[LanguageModelParams]:
        """Get language model parameters."""
        result = await self.widget.send_request("params")

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

        params = {"sessionId": self.session_id, "input": input_data, "options": options_dict}
        result = await self.widget.send_request("prompt", params)

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
        self.widget._stream_chunks[request_id] = []

        params = {
            "sessionId": self.session_id,
            "requestId": request_id,
            "input": input_data,
            "options": options_dict,
        }

        # Start the request
        asyncio.create_task(self.widget.send_request("promptStreaming", params))

        # Yield chunks as they arrive
        chunk_index = 0
        while True:
            chunks = self.widget._stream_chunks.get(request_id, [])
            if chunk_index < len(chunks):
                yield chunks[chunk_index]
                chunk_index += 1
            else:
                # Check if the request is complete
                await asyncio.sleep(0.1)
                if request_id not in self.widget._pending_requests:
                    # Request completed, yield remaining chunks
                    chunks = self.widget._stream_chunks.get(request_id, [])
                    while chunk_index < len(chunks):
                        yield chunks[chunk_index]
                        chunk_index += 1
                    break

        # Clean up
        if request_id in self.widget._stream_chunks:
            del self.widget._stream_chunks[request_id]

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

        params = {"sessionId": self.session_id, "input": input_data, "options": options_dict}
        result = await self.widget.send_request("append", params)

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

        params = {"sessionId": self.session_id, "input": input_data, "options": options_dict}
        result = await self.widget.send_request("measureInputUsage", params)

        return result["usage"]

    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._session_id
    
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
            "sessionId": self.session_id,
            "newSessionId": new_session_id,
            "options": options_dict,
        }
        result = await self.widget.send_request("clone", params)
        newLM = LanguageModel(self.widget)
        newLM._session_id = new_session_id
        newLM._top_k = result.get("topK")
        newLM._temperature = result.get("temperature")
        newLM._input_usage = result.get("inputUsage", 0.0)
        newLM._input_quota = result.get("inputQuota", float("inf"))
        return newLM

    async def destroy(self) -> None:
        """Destroy the session and free resources."""
        params = {"sessionId": self.session_id}
        await self.widget.send_request("destroy", params)

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

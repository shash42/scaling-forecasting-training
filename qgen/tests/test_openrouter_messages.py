import asyncio
import json
from unittest.mock import AsyncMock

import httpx

from qgen.inference import openrouter_inference


def test_string_and_multiturn_prompts_use_existing_openrouter_client(monkeypatch):
    monkeypatch.setattr(openrouter_inference, 'OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '<answer>1</answer>'}}]})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    engine = openrouter_inference.OpenRouterInference('example/model')
    conversation = [{'role': 'user', 'content': 'Find the dates.'},
                    {'role': 'assistant', 'content': 'Initial answer.'},
                    {'role': 'user', 'content': 'Recheck the dates.'}]
    results = asyncio.run(engine.generate(['Return <answer>1</answer>', conversation], batch_size=2))
    assert len(results) == 2
    assert requests[0]['messages'] == [{'role': 'user', 'content': 'Return <answer>1</answer>'}]
    assert requests[1]['messages'] == conversation
    assert all(request['model'] == 'example/model' for request in requests)

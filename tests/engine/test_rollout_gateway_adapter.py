import asyncio
import json

from rllm.experimental.engine.tinker_adapter import create_rollout_handler
from rllm.experimental.rollout.rollout_engine import ModelOutput
from rllm.tools.tool_base import ToolCall


class FakeRolloutEngine:
    model_name = "fake-engine-model"

    def __init__(self):
        self.calls = []

    async def get_model_response(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ModelOutput(
            content="The answer is 4.",
            reasoning="I added the numbers.",
            tool_calls=[ToolCall(name="calculator", arguments={"expression": "2+2"})],
            prompt_ids=[10, 11, 12],
            completion_ids=[20, 21],
            logprobs=[-0.1, -0.2],
            finish_reason="stop",
        )


def test_rollout_handler_maps_gateway_request_and_token_extensions():
    engine = FakeRolloutEngine()
    handler = create_rollout_handler(engine)

    async def run_request():
        return await handler(
            {
                "model": "request-model",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "tools": [{"type": "function", "function": {"name": "calculator"}}],
                "user": "session-123",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_completion_tokens": 32,
            }
        )

    response = asyncio.run(run_request())

    messages, kwargs = engine.calls[0]
    assert messages == [{"role": "user", "content": "What is 2+2?"}]
    assert kwargs["application_id"] == "session-123"
    assert kwargs["max_tokens"] == 32
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["tools"] == [{"type": "function", "function": {"name": "calculator"}}]

    assert response["model"] == "request-model"
    assert response["prompt_token_ids"] == [10, 11, 12]
    assert response["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    choice = response["choices"][0]
    assert choice["token_ids"] == [20, 21]
    assert choice["logprobs"]["content"] == [{"logprob": -0.1}, {"logprob": -0.2}]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "The answer is 4."
    assert choice["message"]["reasoning"] == "I added the numbers."
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "calculator"
    assert json.loads(tool_call["function"]["arguments"]) == {"expression": "2+2"}

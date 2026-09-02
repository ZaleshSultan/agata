"""Regression checks for deterministic commands added from E.V. ideas."""

import json
from unittest.mock import patch

from action_router import route_media
from agent import Agent


class _FailIfCalledLLM:
    def chat(self, *args, **kwargs):
        raise AssertionError("Deterministic media commands must bypass the LLM")


def test_media_command_routes_without_llm() -> None:
    action = route_media("поставь музыку на паузу")
    assert action is not None
    assert action.tool_name == "control_media"
    assert action.arguments == {"action": "play_pause"}

    agent = Agent()
    agent._llm = _FailIfCalledLLM()
    with patch("agent.execute_tool", return_value=json.dumps({"success": True, "action": "play_pause"})) as execute:
        answer = agent.run("поставь музыку на паузу")
    execute.assert_called_once_with("control_media", {"action": "play_pause"})
    assert "Воспроизведение" in answer


def test_bare_play_stays_out_of_media_router() -> None:
    assert route_media("play a game with me") is None

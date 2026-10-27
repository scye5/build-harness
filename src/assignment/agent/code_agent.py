"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])
        
        self.system_prompt = (
            "You are a coding agent tasked with fixing a software issue in a "
            "repository. Inspect the relevant code, implement a focused fix, "
            "and run appropriate tests using the execute tool. Preserve "
            "unrelated changes. Review the resulting git diff and submit the "
            "fix as a git patch. Use send_message to communicate your results "
            "and report any validation failures honestly.\n\n"
            "<system_information>\n"
            + json.dumps(
                {
                    "machine": environment.machine,
                    "release": environment.release,
                    "system": environment.system,
                    "version": environment.version,
                },
                indent=2,
            )
            + "\n</system_information>"
        )
        self.task_prompt = self.task
        
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            self.system_prompt += (
                "\n\nAvailable skills (use invoke_skill to read full instructions):\n"
                + "\n".join(skill["metadata"] for skill in self.skills.values())
            )
        


    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []
        for call in tool_calls:
            try:
                function = call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")

                if name == "execute":
                    command = arguments.get("command")
                    if not isinstance(command, str) and not (
                        isinstance(command, list)
                        and command
                        and all(isinstance(part, str) for part in command)
                    ):
                        raise ValueError("command must be a string or a nonempty list of strings")
                    allowed = {"command", "shell", "cwd", "timeout", "env"}
                    if arguments.keys() - allowed:
                        raise ValueError("Unsupported execute arguments")
                    content = format_tool_output(self.env.execute(**arguments))
                elif name == "invoke_skill":
                    skill_name = arguments.get("name")
                    if not isinstance(skill_name, str) or skill_name not in self.skills:
                        raise ValueError(f"Unknown skill: {skill_name!r}")
                    content = self.skills[skill_name]["content"]
                elif name == "send_message":
                    summary = arguments.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError("summary must be a string")
                    print(summary, flush=True)
                    self.finished = True
                    content = "Message sent to the user."
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except (KeyError, TypeError, ValueError) as exc:
                content = format_tool_output({"error": str(exc)})

            observations.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
        return observations

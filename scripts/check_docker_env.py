"""Run with `.venv/bin/python scripts/check_docker_env.py [--integration]`.

Offline regression checks live here because the assignment forbids changing
tests/. Integration downloads python:3.12 and bootstraps SWE-ReX; no LLM or
Modal calls are made.
"""
import asyncio
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from assignment.env import AssignmentDockerDeployment, Environment


class DockerChecks(unittest.TestCase):
    def deployment(self):
        return AssignmentDockerDeployment(
            image="python:3.12", runtime_timeout=10, deployment_timeout=60,
            startup_timeout=1, install_pipx=False,
        )

    def test_failed_start_removes_container(self):
        deployment = self.deployment()
        with patch.object(deployment, "_pull_image"), patch(
            "assignment.env.subprocess.Popen"
        ), patch("assignment.env._wait_until_alive", AsyncMock(side_effect=TimeoutError)), patch(
            "assignment.env.subprocess.run", return_value=SimpleNamespace(returncode=0)
        ) as remove:
            with self.assertRaises(TimeoutError):
                asyncio.run(deployment.start())
            self.assertEqual(remove.call_args.args[0][:3], ["docker", "rm", "--force"])
            self.assertIsNone(deployment.container_name)

    def test_start_is_local_and_lifetime_is_supervised(self):
        deployment = self.deployment()
        with patch.object(deployment, "_pull_image"), patch(
            "assignment.env.subprocess.Popen"
        ) as launch, patch("assignment.env._wait_until_alive", AsyncMock()):
            asyncio.run(deployment.start())
            command = launch.call_args.args[0]
            self.assertIn(f"127.0.0.1:{deployment._config.port}:8880", command)
            self.assertIn("--rm", command)
            self.assertIn("60", command)
            self.assertIn("timeout=float(sys.argv[1])", command[command.index("-c") + 1])
            self.assertEqual(command[-5:], ["swerex-remote", "--port", "8880", "--auth-token", deployment.runtime._config.auth_token])
        # Execute the actual supervisor to prove lifetime enforcement without
        # requiring a container or waiting for a network bootstrap.
        supervisor = command[command.index("-c") + 1]
        result = subprocess.run(
            [sys.executable, "-c", supervisor, "0.1", sys.executable,
             "-c", "import time; time.sleep(30)"],
            capture_output=True, timeout=5,
        )
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.stderr, b"")

    def test_initialization_failure_cleans_up(self):
        fake = MagicMock()
        fake.start = AsyncMock()
        fake.is_alive = AsyncMock(return_value=True)
        fake.stop = AsyncMock()
        fake.runtime.execute = AsyncMock(return_value=SimpleNamespace(
            stdout="bad platform response", stderr="", exit_code=0,
        ))
        with patch("assignment.env.AssignmentDockerDeployment", return_value=fake):
            with self.assertRaises(ValueError):
                Environment(backend="docker")
        fake.stop.assert_awaited_once()

    def test_command_contract_and_recovery(self):
        env = Environment.__new__(Environment)
        env.backend, env.cwd, env.env_defaults = "docker", "/work", {"A": "default"}
        env.deployment = MagicMock()
        env.deployment.is_alive = AsyncMock(return_value=True)
        execute = env.deployment.runtime.execute = AsyncMock(return_value=SimpleNamespace(
            stdout="out", stderr="err", exit_code=7,
        ))
        result = env.execute(["false"], shell=False, timeout=3, env={"A": "override"})
        self.assertEqual((result["output"], result["returncode"]), ("outerr", 7))
        command = execute.call_args.args[0]
        self.assertEqual((command.cwd, command.timeout, command.env), ("/work", 3, {"A": "override"}))
        self.assertFalse(command.shell)
        execute.side_effect = TimeoutError("command timeout")
        self.assertEqual(env.execute("sleep 10")["returncode"], -1)
        self.assertIsNone(execute.call_args.args[0].timeout)
        env.deployment.is_alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, "no longer running"):
            env.execute("echo next")

    def test_stop_is_repeatable(self):
        deployment = self.deployment()
        deployment._container_name = "owned-test-container"
        with patch("assignment.env.subprocess.run", return_value=SimpleNamespace(returncode=0)) as remove:
            asyncio.run(deployment.stop())
            asyncio.run(deployment.stop())
            remove.assert_called_once()


def integration():
    with Environment(backend="docker", deployment_timeout=600) as env:
        name = env.deployment.container_name
        assert env.is_alive()
        result = env.execute("printf out; printf err >&2; exit 7")
        assert (result["stdout"], result["stderr"], result["returncode"]) == ("out", "err", 7)
        assert env.execute(["pwd"], shell=False, cwd="/tmp")["output"].strip() == "/tmp"
        env.env_defaults["CHECK_VALUE"] = "default"
        assert env.execute("printenv CHECK_VALUE", env={"CHECK_VALUE": "override"})["output"].strip() == "override"
        assert env.execute("sleep 5", timeout=0.1)["returncode"] == -1
        assert env.execute("echo recovered")["returncode"] == 0
    assert not env.is_alive()
    env.stop()
    assert subprocess.run(["docker", "inspect", name], capture_output=True).returncode != 0
    # Start a short-lived container directly to observe expiry even when
    # bootstrapping cannot finish before the deadline.
    short = AssignmentDockerDeployment(
        image="python:3.12", deployment_timeout=1, startup_timeout=3,
        runtime_timeout=1, install_pipx=True,
    )
    try:
        with patch.object(short, "_get_swerex_start_cmd", return_value=["sleep", "30"]):
            try:
                asyncio.run(short.start())
            except TimeoutError:
                pass
            else:
                raise AssertionError("Container without a runtime unexpectedly became ready")
        assert short.container_name is None
    finally:
        asyncio.run(short.stop())
    print("Docker integration checks passed")


if __name__ == "__main__":
    if "--integration" in sys.argv:
        integration()
    else:
        unittest.main()

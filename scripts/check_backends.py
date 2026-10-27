"""Backend regression checks; --integration runs local chess and evaluation.

--swebench additionally evaluates the vendored Django baseline (large download).
No LLM requests or Modal sandboxes are used. Provided tests/ remain untouched.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from assignment import cli, doctor
from assignment.chess_sandbox import ChessSandbox, IllegalMove, SANDBOX_FILES
from assignment.env import AssignmentDockerDeployment
from assignment.eval.harness import EvaluationSpec, TestStatus, evaluate
from assignment.eval.instances import load
from assignment.task import Task

ROOT = Path(__file__).resolve().parents[1]


class BackendChecks(unittest.TestCase):
    def test_application_port_is_local_and_separate(self):
        deployment = AssignmentDockerDeployment(image="python:3.12", runtime_timeout=1,
            deployment_timeout=60, startup_timeout=1, install_pipx=False, published_ports=[8000])
        with patch.object(deployment, "_pull_image"), patch(
            "assignment.env.subprocess.Popen"
        ) as launch, patch("assignment.env._wait_until_alive", AsyncMock()), patch(
            "assignment.env.subprocess.check_output",
            return_value=json.dumps({"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "49123"}]}),
        ):
            import asyncio
            asyncio.run(deployment.start())
            self.assertIn("127.0.0.1:0:8000", launch.call_args.args[0])
            self.assertEqual(deployment.published_ports, {8000: 49123})
        # No real container was launched; don't let the destructor call Docker.
        deployment._container_name = deployment._container_process = deployment._runtime = None

    def test_evaluation_routes_published_image_and_conda(self):
        spec = load("django__django-15368")
        with patch("assignment.eval.harness.Environment") as env, patch(
            "assignment.eval.harness._grade"
        ), patch("assignment.eval.harness.find_testbed_python", return_value="/testbed-python"):
            evaluate(spec, backend="docker", timeout=1800)
            self.assertEqual(env.call_args.kwargs["image"], spec.image)
            self.assertEqual(env.call_args.kwargs["backend"], "docker")
            self.assertEqual(env.call_args.kwargs["conda_env"], "testbed")
            self.assertGreater(env.call_args.kwargs["deployment_timeout"], 1800)

    def test_cli_backend_reaches_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            patch_file = root / "fix.patch"
            patch_file.write_text("a patch")
            for runner, prefix, environment_name in (
                (cli.run_code_agent, [], "Environment"),
                (cli.run_swebench_agent, ["django__django-15368"], "Environment"),
                (cli.run_chess_agent, [], "ChessSandbox"),
            ):
                for backend in ("modal", "docker"):
                    args = ["runner", *prefix, "--backend", backend, "--model", "test",
                            "--trajectory", str(root / "trace.json")]
                    if runner == cli.run_chess_agent:
                        args += ["--patch", str(patch_file), "--result", str(root / "result.json")]
                    else:
                        args += ["--patch-output", str(patch_file)]
                    with patch.object(sys, "argv", args), patch.object(cli, environment_name) as env, patch.object(
                        cli, "build_testbed_image", return_value="image"
                    ) as build, patch.object(cli, "CodeAgent"), patch.object(cli, "ChessAgent") as agent, contextlib.redirect_stdout(io.StringIO()):
                        env.return_value.__enter__.return_value.execute.return_value = {"returncode": 0, "output": "patch"}
                        agent.return_value.last_state = {}
                        runner()
                        self.assertEqual(env.call_args.kwargs["backend"], backend)
                        if runner == cli.run_code_agent:
                            self.assertEqual(build.call_args.kwargs["backend"], backend)
                        if runner == cli.run_swebench_agent:
                            self.assertEqual(env.call_args.kwargs["conda_env"], "testbed")

    def test_evaluation_scripts_forward_backend(self):
        from assignment import eval as evaluation
        from assignment.eval import harness
        for filename, extra, module in (
            ("evaluate.py", ["--task", str(ROOT / "tasks/chess-terminal-move"),
                             "--evaluation", str(ROOT / "tasks/chess-terminal-move/public_tests")], evaluation),
            ("evaluate_swebench.py", ["django__django-15368"], harness),
        ):
            with patch.object(sys, "argv", [filename, *extra, "--backend", "docker"]), patch.object(
                module, "evaluate"
            ) as grade, contextlib.redirect_stdout(io.StringIO()):
                entry = runpy.run_path(str(ROOT / "scripts" / filename))
                entry["main"]()
                self.assertEqual(grade.call_args.kwargs["backend"], "docker")

    def test_docker_doctor_does_not_require_modal(self):
        with patch.object(sys, "argv", ["doctor", "--offline", "--backend", "docker"]), patch.object(
            doctor, "Config", side_effect=AssertionError("Modal credentials consulted")
        ), patch.object(doctor, "load_dotenv"), patch.object(doctor, "verify_source"), patch.object(
            doctor.subprocess, "run", return_value=MagicMock(stdout="linux\n")
        ), patch.dict(os.environ, {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "test",
                                  "OPENAI_BASE_URL": "https://example.invalid"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(doctor.main(), 0)


def integration():
    task = Task.load(ROOT / "tasks/chess-terminal-move")
    with ChessSandbox(task=task, backend="docker") as sandbox:
        name = sandbox.deployment.container_name
        assert sandbox.server_url.startswith("http://127.0.0.1:")
        state = sandbox.state()
        assert state["turn"] == "white" and state["history"] == []
        try:
            sandbox.play("e2e5")
        except IllegalMove:
            pass
        else:
            raise AssertionError("Illegal move accepted")
        assert sandbox.reset()["history"] == []
        simulation = sandbox.client.post("/api/simulate", json={"fen": state["fen"], "move": "e2e4"})
        simulation.raise_for_status()
        assert simulation.json()["applied_move"] == "e2e4"
        for source in SANDBOX_FILES:
            assert sandbox.execute(["test", "-f", f"/opt/assignment/{source.name}"], shell=False)["returncode"] == 0
        bindings = json.loads(subprocess.check_output(
            ["docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", name], text=True))
        assert all(b["HostIp"] == "127.0.0.1" for values in bindings.values() if values for b in values)
    assert subprocess.run(["docker", "inspect", name], capture_output=True).returncode != 0
    spec = EvaluationSpec.load(task.root / "public_tests")
    report = evaluate(spec, task=task, backend="docker")
    assert not report.error, report.summary()
    assert not report.resolved, "Unfixed task must not pass"
    assert all(status == TestStatus.PASSED for status in report.pass_to_pass.values()), report.summary()
    assert all(status == TestStatus.FAILED for status in report.fail_to_pass.values()), report.summary()
    print("Chess localhost API, support files, cleanup, and baseline evaluation passed")


if __name__ == "__main__":
    if "--integration" in sys.argv:
        integration()
    elif "--swebench" in sys.argv:
        report = evaluate(load("django__django-15368"), backend="docker", timeout=1800)
        assert not report.error, report.summary()
        assert all(s == TestStatus.PASSED for s in report.pass_to_pass.values()), report.summary()
        assert all(s == TestStatus.FAILED for s in report.fail_to_pass.values()), report.summary()
        print("Django baseline and testbed Conda environment verified")
    else:
        unittest.main()

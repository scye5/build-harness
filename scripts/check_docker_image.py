"""Offline checks: uv run python scripts/check_docker_image.py

Real chess image build: add --integration (downloads dependencies, no API calls).
Kept outside tests/ as required by the assignment.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from assignment.task import Task
from assignment.utils.image import SourceMismatch, build_testbed_image

IMAGE_ID = "sha256:" + "a" * 64


class ImageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        (source / "app.py").write_text("print('test')\n")
        (source / "nested").mkdir()
        for name in (".env", ".git", "cache.pyc"):
            (source / "nested" / name).write_text("excluded")
        for name in (".venv", ".pytest_cache", "__pycache__"):
            (source / name).mkdir()
            (source / name / "data").write_text("excluded")
        dockerfile = self.root / "Dockerfile"
        dockerfile.write_text("FROM python:3.12-slim\nCOPY . /testbed\n")
        self.task = Task("test", "local", "abc", source, dockerfile, "issue",
                         ("httpx==0.28.1",), self.root)

    def test_filtered_context_pins_and_cache(self):
        contexts = []

        def build(command, **kwargs):
            context = Path(command[-1])
            contexts.append(context)
            files = {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()}
            self.assertEqual(files, {"app.py"})
            last = kwargs["input"].splitlines()[-1]
            self.assertEqual(json.loads(last.removeprefix("RUN ")),
                             ["python", "-m", "pip", "install", "--no-cache-dir", "httpx==0.28.1"])
            self.assertIn("--no-cache", command)
            self.assertEqual(command[command.index("--tag") + 1], "assignment:test")
            Path(command[command.index("--iidfile") + 1]).write_text(IMAGE_ID)

        with patch("assignment.utils.image.verify_source") as verify, patch(
            "assignment.utils.image.subprocess.run", side_effect=build
        ):
            self.assertEqual(build_testbed_image(self.task, backend="docker", force_build=True,
                                                tag="assignment:test"), IMAGE_ID)
            verify.assert_called_once_with(self.task, strict=True)
        self.assertFalse(contexts[0].exists())

    def test_verification_prevents_build(self):
        with patch("assignment.utils.image.verify_source", side_effect=SourceMismatch("dirty")), patch(
            "assignment.utils.image.subprocess.run"
        ) as run:
            with self.assertRaises(SourceMismatch):
                build_testbed_image(self.task, backend="docker")
            run.assert_not_called()

    def test_failure_cleans_context(self):
        contexts = []

        def fail(command, **kwargs):
            contexts.append(Path(command[-1]))
            raise subprocess.CalledProcessError(1, command)

        with patch("assignment.utils.image.verify_source"), patch(
            "assignment.utils.image.subprocess.run", side_effect=fail
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                build_testbed_image(self.task, backend="docker")
        self.assertFalse(contexts[0].exists())

    def test_symlink_cannot_bypass_exclusions(self):
        (self.task.source / "alias").symlink_to(self.task.source / "nested" / ".env")
        with patch("assignment.utils.image.verify_source"), patch(
            "assignment.utils.image.subprocess.run"
        ) as run:
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_testbed_image(self.task, backend="docker")
            run.assert_not_called()

    def test_modal_remains_default_and_pins_last(self):
        image = MagicMock()
        with patch("assignment.utils.image.verify_source") as verify, patch(
            "assignment.utils.image.modal.Image.from_dockerfile", return_value=image
        ):
            result = build_testbed_image(self.task, strict=False)
            verify.assert_called_once_with(self.task, strict=False)
            image.pip_install.assert_called_once_with(*self.task.pins)
            self.assertIs(result, image.pip_install.return_value)


def integration():
    task = Task.load(Path(__file__).resolve().parents[1] / "tasks/chess-terminal-move")
    image = build_testbed_image(task, backend="docker", tag="assignment/chess-terminal-move:local")
    check = """
import importlib.metadata as metadata
import json, pathlib, subprocess, sys
subprocess.run(['swerex-remote', '--help'], check=True, capture_output=True)
assert sys.prefix != '/opt/assignment-swerex', 'Runtime must not replace task Python'
for pin in json.loads(sys.argv[1]):
    name, version = pin.split('==', 1)
    assert metadata.version(name) == version, pin
assert pathlib.Path('/testbed/pyproject.toml').is_file()
assert not pathlib.Path('/testbed/.env').exists()
assert not pathlib.Path('/testbed/.venv').exists()
assert not subprocess.check_output(['git', '-C', '/testbed', 'status', '--porcelain']).strip()
print('All task pins and clean source baseline verified')
"""
    subprocess.run(["docker", "run", "--rm", "--entrypoint", "python", image,
                    "-c", check, json.dumps(task.pins)], check=True)
    print(f"Docker image integration passed: {image}")


if __name__ == "__main__":
    if "--integration" in sys.argv:
        integration()
    else:
        unittest.main()

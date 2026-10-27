import asyncio
import json
import posixpath
import math
import subprocess
import time
from pathlib import PurePath
from typing import Any

import modal
from swerex import __version__ as swerex_version
from swerex.deployment.docker import DockerDeployment
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.free_port import find_free_port
from swerex.utils.wait import _wait_until_alive


class AssignmentDockerDeployment(DockerDeployment):
    """Local SWE-ReX container with bounded lifetime and localhost control port.

    Images must contain python3 and /bin/sh (and bash for shell commands).
    A Python supervisor exits when the lifetime expires, so Docker stops the
    whole container even if the harness process has disappeared.
    """

    def __init__(self, *, runtime_timeout: float, deployment_timeout: float,
                 install_pipx: bool, published_ports: list[int] | None = None, **kwargs: Any):
        if not math.isfinite(deployment_timeout) or deployment_timeout <= 0:
            raise ValueError("deployment_timeout must be finite and positive")
        super().__init__(**kwargs)
        self._runtime_timeout = runtime_timeout
        self._deployment_timeout = deployment_timeout
        self._install_pipx = install_pipx
        self.published_ports = dict.fromkeys(published_ports or [])
        if any(type(port) is not int or not 1 <= port <= 65535 or port == 8880
               for port in self.published_ports):
            raise ValueError("Published ports must be 1..65535, excluding SWE-ReX port 8880")

    def _get_swerex_start_cmd(self, token: str) -> list[str]:
        args = f"--port 8880 --auth-token {token}"
        return ["/bin/sh", "-c", (
            f"swerex-remote {args} || (python3 -m pip install pipx && "
            f"python3 -m pipx run --spec swe-rex=={swerex_version} swerex-remote {args})"
        )]

    async def start(self) -> None:
        if self._container_name is not None:
            raise RuntimeError("Docker deployment is already started")
        self._pull_image()
        self._config.port = self._config.port or find_free_port()
        self._container_name = self._get_container_name()
        token = self._get_token()
        command = (self._get_swerex_start_cmd(token) if self._install_pipx
                   else ["swerex-remote", "--port", "8880", "--auth-token", token])
        supervisor = (
            "import subprocess,sys\n"
            "try:\n"
            "    code = subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode\n"
            "except subprocess.TimeoutExpired:\n"
            "    code = 124\n"
            "sys.exit(code)\n"
        )
        try:
            publish_args = []
            for port in self.published_ports:
                # Docker chooses the host port atomically, avoiding collisions.
                publish_args.extend(["--publish", f"127.0.0.1:0:{port}"])
            # Override any image entrypoint so the supervisor owns PID 1.
            self._container_process = subprocess.Popen(
                ["docker", "run", "--rm", "--publish",
                 f"127.0.0.1:{self._config.port}:8880", *publish_args,
                 "--name", self._container_name,
                 "--entrypoint", "python3", self._config.image,
                 "-c", supervisor, str(self._deployment_timeout), *command],
                # Inherit logs: reading a live container's PIPE on timeout can hang.
            )
            self._runtime = RemoteRuntime(
                host="http://127.0.0.1", port=self._config.port,
                timeout=self._runtime_timeout, auth_token=token,
            )
            await _wait_until_alive(
                self.is_alive, timeout=self._config.startup_timeout,
                function_timeout=min(self._runtime_timeout, 1),
            )
            if self.published_ports:
                bindings = json.loads(subprocess.check_output(
                    ["docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}",
                     self._container_name], text=True, timeout=10,
                ))
                for port in self.published_ports:
                    binding = bindings[f"{port}/tcp"]
                    if len(binding) != 1 or binding[0]["HostIp"] != "127.0.0.1":
                        raise RuntimeError(f"Port {port} is not bound exclusively to localhost")
                    self.published_ports[port] = int(binding[0]["HostPort"])
        except BaseException as exc:
            try:
                await self.stop()
            except Exception as cleanup_error:
                exc.add_note(f"Docker cleanup also failed: {cleanup_error}")
            raise

    async def is_alive(self, *, timeout: float | None = None):
        if (self._runtime is None or self._container_process is None
                or self._container_process.poll() is not None):
            return False
        return await self._runtime.is_alive(timeout=timeout)

    async def stop(self) -> None:
        # Removing the container terminates its runtime and every child command.
        # Do this even when HTTP is broken or startup only partially completed.
        if self._container_name is not None:
            result = await asyncio.to_thread(
                subprocess.run, ["docker", "rm", "--force", self._container_name],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode and "No such container" not in result.stderr:
                raise RuntimeError(f"Docker cleanup failed: {result.stderr.strip()}")
        if self._container_process is not None:
            try:
                await asyncio.to_thread(self._container_process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self._container_process.kill()
                await asyncio.to_thread(self._container_process.wait, timeout=5)
        self._runtime = None
        self._container_process = None
        self._container_name = None


def _tls_port_configuration(
    modal_sandbox_kwargs: dict[str, Any], runtime_port: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Put the SWE-ReX control port on the TLS tunnel exactly once."""

    kwargs = dict(modal_sandbox_kwargs)
    encrypted = list(kwargs.pop("encrypted_ports", []))
    unencrypted = [
        port for port in kwargs.pop("unencrypted_ports", []) if port != runtime_port
    ]
    encrypted = list(dict.fromkeys([runtime_port, *encrypted]))
    return encrypted, unencrypted, kwargs


class AssignmentModalDeployment(ModalDeployment):
    """Modal deployment with a correctly matched TLS tunnel and URL.

    SWE-ReX 1.4.0 opens its control port as unencrypted TCP but initializes the
    HTTP client with ``Tunnel.url``, which is Modal's HTTPS endpoint. Keep the
    upstream runtime and authentication protocol, but expose that control port
    through ``encrypted_ports`` so its URL and transport agree.
    """

    async def start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started; ignoring duplicate start().")
            return

        self.logger.info("Starting Modal sandbox")
        self._hooks.on_custom_step("Starting Modal sandbox")
        started = time.monotonic()
        token = self._get_token()
        encrypted, unencrypted, extra_kwargs = _tls_port_configuration(
            self._modal_kwargs, self._port
        )
        try:
            self._sandbox = await modal.Sandbox.create.aio(
                "/usr/bin/env",
                "bash",
                "-c",
                self._start_swerex_cmd(token),
                image=self._image,
                timeout=int(self._deployment_timeout),
                encrypted_ports=encrypted,
                unencrypted_ports=unencrypted,
                app=self._app,
                **extra_kwargs,
            )
            tunnels = await self._sandbox.tunnels.aio()
            tunnel = tunnels[self._port]
            elapsed = time.monotonic() - started
            log_url = await self.get_modal_log_url()
            self.logger.info("Sandbox %s created in %.2fs", self._sandbox.object_id, elapsed)
            self.logger.info("Modal logs: %s", log_url)
            await asyncio.sleep(1)
            self.logger.info("Starting SWE-ReX runtime at %s", tunnel.url)
            self._hooks.on_custom_step("Starting runtime")
            self._runtime = RemoteRuntime(
                host=tunnel.url,
                timeout=self._runtime_timeout,
                auth_token=token,
                logger=self.logger,
            )
            remaining = max(0, self._startup_timeout - elapsed)
            await self._wait_until_alive(timeout=remaining)
        except Exception as exc:
            details = [f"{type(exc).__name__}: {exc}"]
            if self._sandbox is not None:
                details.append(f"sandbox_id={self._sandbox.object_id}")
                try:
                    details.append(f"logs={await self.get_modal_log_url()}")
                except Exception:
                    pass
                try:
                    await self._sandbox.terminate.aio()
                except Exception:
                    pass
            self._runtime = None
            self._sandbox = None
            raise RuntimeError(
                "Modal sandbox started, but the SWE-ReX control runtime did not "
                "become reachable. " + " | ".join(details)
            ) from exc

class Environment:
    """
    Executes bash commands in a Modal or local Docker sandbox via SWE-ReX.
    """

    # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py

    def __init__(
        self,
        image: "str | PurePath | modal.Image" = "python:3.12",
        cwd: str = "/",
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        install_pipx: bool = True,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        conda_env: str | None = None,
        backend: str = "modal",
        published_ports: list[int] | None = None,
    ):
        """Launch a sandbox and block until its runtime answers.

        Args:
            image: A prebuilt `modal.Image`, a Dockerhub or ECR image name, or a
                path to a Dockerfile. Build a task testbed with
                `assignment.utils.image.build_testbed_image`, which is the only
                way to pass a credential to a private clone.
            cwd: Working directory for commands that do not specify one.
            startup_timeout: Seconds to wait for the SWE-ReX runtime to come up.
            runtime_timeout: Timeout for SWE-ReX health requests. Command limits
                are supplied separately to execute(timeout=...).
            deployment_timeout: Maximum sandbox lifetime in seconds.
            install_pipx: Install pipx in the image, needed to bootstrap the
                SWE-ReX server when it is not already present.
            modal_sandbox_kwargs: Additional keyword arguments forwarded to
                ``modal.Sandbox.create``. This is used for capabilities such as
                encrypted port forwarding.
            conda_env: Name of a conda environment to put on PATH for every
                command. SWE-bench images install the repository under test into
                an environment named ``testbed`` but never activate it, so
                without this ``python`` is conda's base environment, where the
                repository and its dependencies are not installed.
            backend: "modal" (default) or "docker". Docker requires a prebuilt
                image name, with python3, /bin/sh, and bash. install_pipx=False
                requires swerex-remote already installed in the image. Docker
                does not accept Modal images, Dockerfile paths, or Modal kwargs.
            published_ports: Docker container ports to publish on automatically
                allocated localhost ports. Use tunnel_url() to get their URLs.
        """
        if backend not in {"modal", "docker"}:
            raise ValueError(f"Unknown environment backend: {backend!r}")
        self.backend = backend
        if backend == "docker":
            for name, value in (("startup_timeout", startup_timeout),
                                ("runtime_timeout", runtime_timeout)):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        self.cwd = cwd
        # Merged into every command's environment; a per-call `env` wins.
        self.env_defaults: dict[str, str] = {}
        deployment_kwargs = dict(
            image=image,
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            install_pipx=install_pipx,
        )
        if backend == "docker":
            if not isinstance(image, str) or not image.strip():
                raise TypeError("Docker requires a nonempty prebuilt image name")
            if modal_sandbox_kwargs:
                raise ValueError("modal_sandbox_kwargs cannot be used with Docker")
            self.deployment = AssignmentDockerDeployment(
                **deployment_kwargs, published_ports=published_ports,
            )
        else:
            if published_ports:
                raise ValueError("published_ports is only supported with Docker")
            self.deployment = AssignmentModalDeployment(
                **deployment_kwargs, modal_sandbox_kwargs=modal_sandbox_kwargs,
            )

        async def _start():
            await self.deployment.start()
            if not await self.deployment.is_alive():
                raise RuntimeError("Sandbox runtime did not become ready")

        try:
            asyncio.run(_start())
            if conda_env:
                self.activate_conda_env(conda_env)
            # Read the platform inside the sandbox, not on the harness host.
            self.system, self.release, self.version, self.machine = self.execute(
                "uname -s; uname -r; uname -v; uname -m"
            )["output"].splitlines()
        except BaseException as exc:
            try:
                self.stop()
            except Exception as cleanup_error:
                exc.add_note(f"Sandbox cleanup also failed: {cleanup_error}")
            raise

    def is_alive(self) -> bool:
        """Whether the sandbox is still running.

        Modal reclaims a sandbox once `deployment_timeout` elapses, and the
        deployment keeps its handle afterwards, so this asks the sandbox itself
        rather than trusting the handle's existence.
        """
        if self.backend == "docker":
            try:
                return bool(asyncio.run(self.deployment.is_alive(timeout=2)))
            except Exception:
                return False
        sandbox = self.deployment._sandbox
        return sandbox is not None and sandbox.poll() is None

    def activate_conda_env(self, name: str, root: str = "/opt/miniconda3") -> str:
        """Put a conda environment's bin directory first on PATH for all commands.

        Activating properly needs a login shell, which these commands do not
        get. Prepending the environment's bin directory has the same effect for
        ``python``, ``pip``, and anything else installed there.

        Args:
            name: The environment name, for example ``testbed``.
            root: Where conda is installed in the image.

        Returns:
            The PATH now used for every command.

        Raises:
            FileNotFoundError: If the environment is not in the image, rather
                than silently leaving the wrong interpreter on PATH.
        """
        binary_dir = posixpath.join(root, "envs", name, "bin")
        if self.execute(f"test -d {binary_dir}", cwd="/")["returncode"] != 0:
            raise FileNotFoundError(f"No conda environment at {binary_dir}")

        current = self.execute("printenv PATH", cwd="/")["output"].strip()
        self.env_defaults["PATH"] = f"{binary_dir}:{current}"
        return self.env_defaults["PATH"]

    def execute(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        # Command defaults to shell=False, which requires `command` to be an argv
        # list; default to shell=True so a plain command string works.
        shell: bool | None = True,
        # check=False so a failing command returns its exit code instead of
        # raising and discarding the output the agent needs to see.
        check: bool = False,
    ) -> dict:
        """Run a command in the sandbox and return its result.

        Args:
            command: A shell string, or an argv list if `shell` is False.
            timeout: Seconds before the command is killed. None means no timeout.
            cwd: Working directory to run the command in. Defaults to the
                environment's `cwd`.
            env: Environment variables to set for the command.
            shell: Run the command through a shell, as with `subprocess.run`.
                None is treated as True, since a model calling this as a tool may
                send null for an omitted argument.
            check: Raise on a non-zero exit code instead of reporting it.

        Returns:
            A dict with `output`, `returncode`, and `exception_info`. `output`
            holds stdout followed by stderr. `exception_info` is empty when the command ran, whatever
            its exit code. If the call itself fails (the sandbox died, the
            command timed out), the exception is caught rather than raised:
            `returncode` is -1, `exception_info` describes it, and an `extra` key
            carries the exception type.
        """
        # Command.shell is a strict bool, so None would fail validation.
        shell = True if shell is None else shell

        arguments = {
            "command": command,
            "timeout": timeout,
            "env": {**self.env_defaults, **(env or {})} or None,
            "shell": shell,
            "check": check,
            "merge_output_streams": False,
        }
        if cwd is not None:
            arguments["cwd"] = cwd
        # A cwd from the caller wins; otherwise the environment's default applies.
        arguments.setdefault("cwd", self.cwd)

        try:
            result = asyncio.run(self.deployment.runtime.execute(Command(**arguments)))
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                # The coding agent consumes one observation string, while the
                # task harness needs the streams separately for diagnostics.
                "output": result.stdout,
                "returncode": result.exit_code,
                "exception_info": "",
            }
            if result.stderr:
                output["output"] += result.stderr
        except Exception as e:
            # A command can fail for reasons worth reporting to the model, but a
            # sandbox that is gone is terminal: every later command fails the
            # same way, so raise rather than let the caller keep going.
            if not self.is_alive():
                raise RuntimeError(
                    "The sandbox is no longer running, so no further commands can "
                    f"be executed. Last error: {e}"
                ) from e

            # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/extra/swerex_modal.py#L82-L87
            # Same keys as the success path, so callers never have to branch on
            # which one they got.
            output = {
                "stdout": "",
                "stderr": str(e) if str(e) else "",
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        return output

    def stop(self, timeout: float = 10):
        """Shut down the runtime and terminate the sandbox.

        Args:
            timeout: Seconds allowed for each of the shutdown and the
                termination steps.
        """

        async def _stop():
            if self.backend == "docker":
                await asyncio.wait_for(self.deployment.stop(), timeout=timeout)
                return
            # ModalDeployment.stop() has an inverted poll() check and only
            # terminates sandboxes that have *already* exited, so a live sandbox
            # leaks until deployment_timeout. Terminate it explicitly. Grab the
            # reference first: stop() clears _sandbox.
            sandbox = self.deployment._sandbox
            try:
                await asyncio.wait_for(self.deployment.stop(), timeout=timeout)
            finally:
                if sandbox is not None:
                    await asyncio.wait_for(sandbox.terminate.aio(), timeout=timeout)

        asyncio.run(_stop())

    def tunnel_url(self, port: int) -> str:
        """Return the public URL for a port forwarded when the sandbox started."""

        if self.backend == "docker":
            host_port = self.deployment.published_ports.get(port)
            if host_port is None or self.deployment.container_name is None:
                raise ValueError(f"Port {port} is not published by a running deployment")
            return f"http://127.0.0.1:{host_port}"

        async def _tunnel_url() -> str:
            tunnels = await self.deployment.sandbox.tunnels.aio()
            try:
                return tunnels[port].url
            except KeyError as exc:
                forwarded = ", ".join(str(item) for item in sorted(tunnels)) or "none"
                raise ValueError(
                    f"Port {port} was not forwarded. Available forwarded ports: {forwarded}."
                ) from exc

        return asyncio.run(_tunnel_url())

    def __enter__(self):
        """Enter a `with` block. The sandbox is already running by this point."""
        return self

    def __exit__(self, *exc):
        """Terminate the sandbox on leaving a `with` block, including on error."""
        self.stop()

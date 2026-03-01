from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
PROJECT_NAME = "miniflow-pr-validate"
API_URL = "http://localhost:8000"
REQUIRED_S2S_KEYS = {
    "transcript",
    "response",
    "audio",
    "sample_rate",
    "request_id",
    "latency_ms",
    "release_id",
}


def run(cmd: Iterable[str], cwd: Path = ROOT_DIR, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command and raise on non-zero exit."""
    return subprocess.run(
        list(cmd),
        cwd=cwd.as_posix(),
        check=True,
        text=True,
        capture_output=capture,
    )


def compose_cmd(*args: str) -> list[str]:
    """Build a docker compose command with fixed project/file context."""
    return [
        "docker",
        "compose",
        "-p",
        PROJECT_NAME,
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]


def cleanup() -> None:
    """Tear down validation containers/volumes, ignoring cleanup errors."""
    try:
        subprocess.run(
            compose_cmd("down", "-v", "--remove-orphans"),
            cwd=str(ROOT_DIR),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    try:
        # Also tear down default compose project to avoid port-8000 collisions.
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v", "--remove-orphans"],
            cwd=str(ROOT_DIR),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _check_endpoint_ok(path: str) -> None:
    """Raise when endpoint does not respond with HTTP 200."""
    with urlopen(f"{API_URL}{path}", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")


def _read_json_endpoint(path: str) -> dict:
    """Return JSON response from endpoint and validate it is an object."""
    with urlopen(f"{API_URL}{path}", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} returned non-object JSON payload")
        return payload


def wait_for_health(timeout_seconds: int = 300, interval_seconds: float = 2.0) -> float:
    """Poll /health until ready or timeout with robust transient-error handling.

    Returns the remaining timeout seconds for use by subsequent checks.
    """
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            _check_endpoint_ok("/health")
            return max(0, deadline - time.time())
        except (
            URLError,
            HTTPError,
            ConnectionResetError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            last_error = exc

        ps_output = run(compose_cmd("ps"), capture=True).stdout
        if "Exit" in ps_output or "Exited" in ps_output:
            logs_output = run(compose_cmd("logs", "--tail=200", "api"), capture=True).stdout
            raise RuntimeError(
                "API container exited during startup.\n"
                f"docker compose ps:\n{ps_output}\n\n"
                f"docker compose logs:\n{logs_output}"
            )

        time.sleep(interval_seconds)

    logs_output = run(compose_cmd("logs", "--tail=200", "api"), capture=True).stdout
    raise RuntimeError(
        f"/health did not become ready within {timeout_seconds} seconds. "
        f"Last error: {last_error}\n\n"
        f"docker compose logs:\n{logs_output}"
    )


def wait_for_ready(remaining_timeout: float, interval_seconds: float = 2.0) -> None:
    """Poll /ready until ready or timeout with robust transient-error handling.

    Uses remaining_timeout to ensure total wait time doesn't exceed original budget.
    """
    deadline = time.time() + remaining_timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            payload = _read_json_endpoint("/ready")
            if payload.get("status") != "ready":
                raise RuntimeError(f"/ready status not ready: {payload}")
            for required_key in ("status", "cuda_available", "device"):
                if required_key not in payload:
                    raise RuntimeError(f"/ready missing required key: {required_key}")
            return
        except (
            URLError,
            HTTPError,
            ConnectionResetError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc

        ps_output = run(compose_cmd("ps"), capture=True).stdout
        if "Exit" in ps_output or "Exited" in ps_output:
            logs_output = run(compose_cmd("logs", "--tail=200", "api"), capture=True).stdout
            raise RuntimeError(
                "API container exited during readiness check.\n"
                f"docker compose ps:\n{ps_output}\n\n"
                f"docker compose logs:\n{logs_output}"
            )

        time.sleep(interval_seconds)

    logs_output = run(compose_cmd("logs", "--tail=200", "api"), capture=True).stdout
    raise RuntimeError(
        f"/ready did not become ready within {remaining_timeout:.1f} seconds. "
        f"Last error: {last_error}\n\n"
        f"docker compose logs:\n{logs_output}"
    )


def pick_sample_path(explicit_path: str | None) -> Path:
    """Return explicit sample path or first WAV under data_assets/test_samples."""
    if explicit_path:
        sample = Path(explicit_path)
        if not sample.exists():
            raise FileNotFoundError(f"E2E sample does not exist: {sample}")
        return sample

    sample_dir = ROOT_DIR / "data_assets" / "test_samples"
    candidates = sorted(sample_dir.glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(
            "No .wav sample found in data_assets/test_samples. "
            "Set E2E_SAMPLE_PATH to an explicit file."
        )
    return candidates[0]


def run_s2s_sample(sample_path: Path, timeout_seconds: int) -> None:
    """Send one sample to /s2s and validate required response keys."""
    response_file = ROOT_DIR / ".tmp_validate_pr_s2s.json"
    try:
        run(
            [
                "curl",
                "-fsS",
                "--max-time",
                str(timeout_seconds),
                "-X",
                "POST",
                f"{API_URL}/s2s",
                "-F",
                f"audio_file=@{sample_path};type=audio/wav",
                "-o",
                str(response_file),
            ]
        )
        payload = json.loads(response_file.read_text())
        missing = REQUIRED_S2S_KEYS - payload.keys()
        if missing:
            raise RuntimeError(f"/s2s response missing keys: {sorted(missing)}")
    finally:
        if response_file.exists():
            response_file.unlink()


def main() -> int:
    """Run PR validation pipeline (tests + container + optional /s2s e2e check)."""
    parser = argparse.ArgumentParser(description="Validate MiniFlow PR changes end-to-end.")
    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        help="Skip /s2s single-sample end-to-end request.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600")),
        help="Timeout for /s2s end-to-end request.",
    )
    parser.add_argument(
        "--e2e-sample-path",
        type=str,
        default=os.getenv("E2E_SAMPLE_PATH"),
        help="Optional WAV file path for /s2s check. Defaults to first WAV under data_assets/test_samples.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=int,
        default=int(os.getenv("STARTUP_TIMEOUT_SECONDS", "300")),
        help="Timeout for /health and /ready startup checks.",
    )
    args = parser.parse_args()

    os.chdir(ROOT_DIR)
    try:
        # Ensure no stale service from a previous run keeps port 8000 occupied.
        cleanup()

        print("[1/5] Running pytest suite...")
        run(["uv", "run", "python", "-m", "pytest", "-q"])

        print("[2/5] Verifying docker compose config...")
        run(compose_cmd("config"), capture=True)

        print("[3/5] Building API container image...")
        run(compose_cmd("build", "api"))

        print("[4/5] Starting API container...")
        run(compose_cmd("up", "-d", "api"))

        print("[5/5] Checking /health...")
        remaining_timeout = wait_for_health(timeout_seconds=args.startup_timeout_seconds)
        print("Container health check passed.")

        print("Checking /ready...")
        wait_for_ready(remaining_timeout=remaining_timeout)
        print("Container readiness check passed.")

        # Safety check: verify CUDA works inside container before /s2s test
        print("Verifying CUDA functionality...")
        try:
            result = run(
                [
                    "curl",
                    "-fsS",
                    "-X",
                    "GET",
                    f"{API_URL}/ready",
                ],
                capture=True,
            )
            import json
            ready_info = json.loads(result.stdout)
            if ready_info.get("device") != "cuda":
                raise RuntimeError(
                    f"CUDA not available or disabled. Device: {ready_info.get('device')}. "
                    "Set MINIFLOW_DEVICE=cpu or fix CUDA issues."
                )
            # Additional CUDA sanity check - run a simple tensor operation
            run(
                [
                    "docker", "exec", "miniflow-pr-validate-api-1",
                    "/app/.venv/bin/python", "-c",
                    "import torch; x = torch.randn(2,2).cuda(); y = x @ x.T; print('CUDA OK')"
                ],
                capture=True,
            )
            print("CUDA functionality verified.")
        except Exception as e:
            raise RuntimeError(
                f"CUDA safety check failed: {e}\n"
                "The /s2s endpoint requires working CUDA. "
                "Either fix CUDA issues or use --skip-e2e flag."
            )

        if args.skip_e2e:
            print("Skipping /s2s end-to-end check (--skip-e2e).")
        else:
            sample = pick_sample_path(args.e2e_sample_path)
            print(f"Running /s2s end-to-end check with sample: {sample}")
            run_s2s_sample(sample, timeout_seconds=args.request_timeout_seconds)
            print("/s2s end-to-end check passed.")
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())

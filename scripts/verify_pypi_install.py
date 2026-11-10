"""Verify an actual PyPI wheel install in a new virtual environment."""

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import venv


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_pypi_install.py VERSION")
    version = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="neurify-pypi-") as directory:
        root = Path(directory)
        venv.create(root, with_pip=True)
        python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        command = [str(python), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir",
                   "--index-url", "https://pypi.org/simple/", f"neurify=={version}"]
        for attempt in range(6):
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                break
            if attempt == 5:
                print(result.stdout, result.stderr)
                raise SystemExit(result.returncode)
            time.sleep(10)
        script = Path(__file__).with_name("smoke_install.py")
        subprocess.run([str(python), "-I", str(script), "--expected-version", version], check=True)


if __name__ == "__main__":
    main()

"""Regression tests for production importability.

`palm_9000/utils.py` and `palm_9000/adc0834.py` used to import dev-only
packages at module scope, so neither could be loaded under
`uv run --no-dev`. That blocked the soil-moisture sensor, which needs
ADC0834 in production.

These tests fail if either regression is reintroduced.
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pyproject() -> dict:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter, without this suite's conftest fakes."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "GOOGLE_API_KEY": "packaging-test-placeholder",
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )


# Blocking a name in sys.modules with None makes `import name` raise
# ImportError, which simulates the package being absent.
BLOCK_OPTIONAL = """
import sys
for name in ("scipy", "scipy.signal", "sounddevice", "pyaudio"):
    sys.modules[name] = None
"""


class TestUtilsImportsWithoutDevDependencies:
    def test_module_imports_with_optional_packages_absent(self):
        result = run_isolated(
            BLOCK_OPTIONAL + "import palm_9000.utils\nprint('IMPORT_OK')\n"
        )
        assert "IMPORT_OK" in result.stdout, (
            "palm_9000.utils must import without scipy/sounddevice/pyaudio.\n"
            f"stderr:\n{result.stderr}"
        )

    def test_dependency_free_helper_still_works(self):
        result = run_isolated(
            BLOCK_OPTIONAL + "from palm_9000.utils import remove_whitespace\n"
            "print(remove_whitespace(' a b '))\n"
        )
        assert result.stdout.strip().endswith("ab"), result.stderr

    def test_function_needing_a_missing_package_raises_clearly(self):
        """Only the individual function should fail, and with ImportError."""
        result = run_isolated(
            BLOCK_OPTIONAL + "import palm_9000.utils as u\n"
            "try:\n"
            "    u.wait_until_device_available(1)\n"
            "except ImportError:\n"
            "    print('RAISED_IMPORT_ERROR')\n"
        )
        assert "RAISED_IMPORT_ERROR" in result.stdout, result.stderr

    def test_no_optional_imports_at_module_scope(self):
        """The imports must sit inside functions, not at the top of the file.

        Checked via the AST rather than text, so prose in the docstring that
        merely names these packages does not trip it.
        """
        source = (PROJECT_ROOT / "palm_9000" / "utils.py").read_text()
        tree = ast.parse(source)

        top_level = set()
        for node in tree.body:  # module scope only, not nested function bodies
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])

        offenders = top_level & {"scipy", "sounddevice", "pyaudio"}
        assert not offenders, (
            f"{sorted(offenders)} imported at module scope again; defer into "
            "the function that uses it so the module loads under --no-dev."
        )


class TestAdc0834IsAProductionDependency:
    def test_rpi_gpio_is_a_main_dependency(self):
        """ADC0834 ships in production, so its GPIO library must too."""
        main_deps = pyproject()["project"]["dependencies"]
        assert any("rpi-gpio" in dep for dep in main_deps), (
            "rpi-gpio must be a main dependency; palm_9000/adc0834.py imports "
            "RPi.GPIO and is needed by the soil-moisture sensor in production."
        )

    def test_rpi_gpio_is_not_only_in_the_dev_group(self):
        dev_group = [
            dep
            for dep in pyproject()["dependency-groups"]["dev"]
            if isinstance(dep, str)
        ]
        assert not any("rpi-gpio" in dep for dep in dev_group), (
            "rpi-gpio should live in [project.dependencies], not the dev group."
        )

    def test_rpi_gpio_is_marked_linux_only(self):
        """The wheel does not build on macOS, so it needs a platform marker."""
        main_deps = pyproject()["project"]["dependencies"]
        dep = next(d for d in main_deps if "rpi-gpio" in d)
        assert "sys_platform" in dep and "linux" in dep, dep

    def test_matches_the_spidev_precedent(self):
        """spidev is the existing Linux-only hardware dependency; mirror it."""
        main_deps = pyproject()["project"]["dependencies"]
        spidev = next(d for d in main_deps if "spidev" in d)
        rpi = next(d for d in main_deps if "rpi-gpio" in d)
        assert spidev.split(";")[1].strip() == rpi.split(";")[1].strip()

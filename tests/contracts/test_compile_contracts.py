from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONTRACT_ROOT = PROJECT_ROOT / "tests" / "contracts"
MAPPED_READER_ROOT = CONTRACT_ROOT / "mapped_reader"
PUBLIC_API_ROOT = CONTRACT_ROOT / "public_api"

NEGATIVE_CONTRACTS = {
    "negative/owner_copy.mojo": (
        "error: 'MappedSafeTensorFile' value has no attribute 'copy'"
    ),
    "negative/mutable_span.mojo": "error: expression must be mutable in assignment",
    "negative/use_after_owner_consume.mojo": (
        "error: use of uninitialized value 'archive'"
    ),
    "negative/escape_owner.mojo": (
        "error: cannot implicitly convert "
        "'Span[UInt8, origin_of(archive)]' value to "
        "'Span[UInt8, ImmStaticOrigin]'"
    ),
}


class CompileContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mojo = shutil.which("mojo")
        if mojo is None:
            raise unittest.SkipTest("the Mojo compiler is not available on PATH")
        cls.mojo = mojo

    def compile_fixture(
        self,
        fixture_path: Path,
        output_directory: Path,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        cache_root = PROJECT_ROOT / ".pixi" / "mojo-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        environment.setdefault("MODULAR_CACHE_DIR", str(cache_root))

        return subprocess.run(
            [
                self.mojo,
                "build",
                "--Werror",
                "-I",
                str(SOURCE_ROOT),
                str(fixture_path),
                "-o",
                str(output_directory / fixture_path.stem),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

    def test_positive_contract_compiles(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="safetensors-mojo-compile-contract-"
        ) as raw_directory:
            completed = self.compile_fixture(
                MAPPED_READER_ROOT / "positive.mojo",
                Path(raw_directory),
            )

        diagnostics = completed.stdout + completed.stderr
        self.assertEqual(
            completed.returncode,
            0,
            msg="positive mapped-reader contract failed to compile:\n"
            + diagnostics,
        )

    def test_negative_contracts_fail_for_the_expected_reason(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="safetensors-mojo-compile-contract-"
        ) as raw_directory:
            output_directory = Path(raw_directory)
            for fixture_name, expected_diagnostic in NEGATIVE_CONTRACTS.items():
                with self.subTest(fixture=fixture_name):
                    completed = self.compile_fixture(
                        MAPPED_READER_ROOT / fixture_name,
                        output_directory,
                    )
                    diagnostics = completed.stdout + completed.stderr
                    self.assertNotEqual(
                        completed.returncode,
                        0,
                        msg=f"negative contract {fixture_name} unexpectedly compiled",
                    )
                    self.assertIn(expected_diagnostic, diagnostics)

    def test_root_public_api_contract_compiles(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="safetensors-mojo-compile-contract-"
        ) as raw_directory:
            completed = self.compile_fixture(
                PUBLIC_API_ROOT / "positive.mojo",
                Path(raw_directory),
            )

        diagnostics = completed.stdout + completed.stderr
        self.assertEqual(
            completed.returncode,
            0,
            msg="root public API contract failed to compile:\n" + diagnostics,
        )


if __name__ == "__main__":
    unittest.main()

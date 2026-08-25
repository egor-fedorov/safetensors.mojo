from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERRORS_PATH = PROJECT_ROOT / "src" / "safetensors" / "errors.mojo"
MEMBER = re.compile(r"^\s+comptime ([A-Z0-9_]+) = Self\(([0-9]+)\)$", re.MULTILINE)


class ErrorKindTests(unittest.TestCase):
    def members(self) -> dict[str, int]:
        return {
            name: int(ordinal)
            for name, ordinal in MEMBER.findall(ERRORS_PATH.read_text(encoding="utf-8"))
        }

    def test_every_error_kind_has_a_production_use(self) -> None:
        production = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((PROJECT_ROOT / "src" / "safetensors").rglob("*.mojo"))
            if path != ERRORS_PATH
        )

        for name in self.members():
            with self.subTest(name=name):
                self.assertIn(f"SafeTensorErrorKind.{name}", production)

    def test_error_ordinals_are_unique_and_retired_values_stay_retired(self) -> None:
        ordinals = list(self.members().values())
        self.assertEqual(len(ordinals), len(set(ordinals)))
        self.assertNotIn(22, ordinals)


if __name__ == "__main__":
    unittest.main()

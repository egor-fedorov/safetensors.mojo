from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from tools.release import prefix_preflight


class PrefixPreflightTests(unittest.TestCase):
    def test_sha256_digest_reads_the_complete_file(self) -> None:
        contents = (b"safetensors-mojo\x00" * 100_000) + b"tail"
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "artifact.conda"
            path.write_bytes(contents)
            self.assertEqual(
                prefix_preflight.sha256_digest(path),
                hashlib.sha256(contents).hexdigest(),
            )

    @patch("tools.release.prefix_preflight.urllib.request.urlopen")
    def test_fetch_repodata_uses_stable_url_headers_and_timeout(
        self,
        urlopen: MagicMock,
    ) -> None:
        expected = {
            "info": {"subdir": "linux-64"},
            "packages.conda": {"artifact.conda": {"sha256": "a" * 64}},
        }
        response = MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(expected).encode())
        urlopen.return_value = response

        actual = prefix_preflight.fetch_repodata(
            "egor-fedorov/safetensors-mojo",
            "linux-64",
        )

        self.assertEqual(actual, expected)
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://prefix.dev/egor-fedorov/safetensors-mojo/"
            "linux-64/repodata.json",
        )
        self.assertEqual(
            request.get_header("User-agent"),
            prefix_preflight.USER_AGENT,
        )
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(request.get_header("Cache-control"), "no-cache")
        self.assertEqual(
            urlopen.call_args.kwargs,
            {"timeout": 30},
        )

    @patch("tools.release.prefix_preflight.urllib.request.urlopen")
    def test_fetch_repodata_returns_empty_only_for_404(
        self,
        urlopen: MagicMock,
    ) -> None:
        urlopen.side_effect = urllib.error.HTTPError(
            "https://prefix.dev/channel/linux-64/repodata.json",
            404,
            "Not Found",
            {},
            None,
        )
        self.addCleanup(urlopen.side_effect.close)
        self.assertIsNone(
            prefix_preflight.fetch_repodata("channel", "linux-64"),
        )

    @patch("tools.release.prefix_preflight.urllib.request.urlopen")
    def test_fetch_repodata_rejects_invalid_successful_documents(
        self,
        urlopen: MagicMock,
    ) -> None:
        invalid_documents = (
            {},
            {"info": {"subdir": "noarch"}, "packages.conda": {}},
            {"info": {"subdir": "linux-64"}, "packages.conda": []},
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                response = MagicMock()
                response.__enter__.return_value = io.BytesIO(
                    json.dumps(document).encode()
                )
                urlopen.return_value = response
                with self.assertRaises(ValueError):
                    prefix_preflight.fetch_repodata("channel", "linux-64")

    @patch("tools.release.prefix_preflight.urllib.request.urlopen")
    def test_fetch_repodata_propagates_403(
        self,
        urlopen: MagicMock,
    ) -> None:
        forbidden = urllib.error.HTTPError(
            "https://prefix.dev/channel/linux-64/repodata.json",
            403,
            "Forbidden",
            {},
            None,
        )
        self.addCleanup(forbidden.close)
        urlopen.side_effect = forbidden
        with self.assertRaises(urllib.error.HTTPError) as raised:
            prefix_preflight.fetch_repodata("channel", "linux-64")
        self.assertIs(raised.exception, forbidden)

    @patch("tools.release.prefix_preflight.urllib.request.urlopen")
    def test_fetch_repodata_propagates_network_errors(
        self,
        urlopen: MagicMock,
    ) -> None:
        failure = urllib.error.URLError("network unavailable")
        urlopen.side_effect = failure
        with self.assertRaises(urllib.error.URLError) as raised:
            prefix_preflight.fetch_repodata("channel", "linux-64")
        self.assertIs(raised.exception, failure)

    def test_find_package_record_searches_current_and_legacy_sections(self) -> None:
        current = {"sha256": "a" * 64}
        legacy = {"sha256": "b" * 64}
        self.assertIs(
            prefix_preflight.find_package_record(
                {"packages.conda": {"current.conda": current}},
                "current.conda",
            ),
            current,
        )
        self.assertIs(
            prefix_preflight.find_package_record(
                {"packages": {"legacy.tar.bz2": legacy}},
                "legacy.tar.bz2",
            ),
            legacy,
        )
        self.assertIsNone(
            prefix_preflight.find_package_record(
                {"packages.conda": {}, "packages": {}},
                "missing.conda",
            )
        )

    def test_find_package_record_rejects_duplicates(self) -> None:
        repodata = {
            "packages.conda": {"artifact.conda": {"sha256": "a" * 64}},
            "packages": {"artifact.conda": {"sha256": "a" * 64}},
        }
        with self.assertRaisesRegex(ValueError, "more than once"):
            prefix_preflight.find_package_record(repodata, "artifact.conda")

    def test_find_package_record_rejects_invalid_repodata(self) -> None:
        invalid_values = (
            [],
            {},
            {"packages.conda": []},
            {"packages": {"artifact.conda": []}},
        )
        for repodata in invalid_values:
            with self.subTest(repodata=repodata):
                with self.assertRaises(ValueError):
                    prefix_preflight.find_package_record(
                        repodata,
                        "artifact.conda",
                    )

    def test_decide_publish_allows_an_absent_record(self) -> None:
        self.assertTrue(prefix_preflight.decide_publish(None, "a" * 64))

    def test_decide_publish_skips_an_identical_record(self) -> None:
        self.assertFalse(
            prefix_preflight.decide_publish(
                {"sha256": "A" * 64},
                "a" * 64,
            )
        )

    def test_decide_publish_rejects_different_or_missing_digests(self) -> None:
        with self.assertRaisesRegex(ValueError, "different SHA-256"):
            prefix_preflight.decide_publish({"sha256": "b" * 64}, "a" * 64)
        with self.assertRaisesRegex(ValueError, "has no SHA-256"):
            prefix_preflight.decide_publish({}, "a" * 64)
        with self.assertRaisesRegex(ValueError, "invalid SHA-256"):
            prefix_preflight.decide_publish({"sha256": "not-a-digest"}, "a" * 64)
        with self.assertRaisesRegex(ValueError, "invalid SHA-256"):
            prefix_preflight.decide_publish(None, "not-a-digest")

    def test_write_github_output_appends_exact_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "github-output"
            path.write_text("existing=value\n", encoding="utf-8")
            prefix_preflight.write_github_output(path, True)
            prefix_preflight.write_github_output(path, False)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "existing=value\npublish=true\npublish=false\n",
            )

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_cli_uses_default_subdir_and_writes_publish_output(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        fetch_repodata.return_value = None
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            artifact = directory / "safetensors-mojo-0.2.0-build.conda"
            artifact.write_bytes(b"artifact")
            github_output = directory / "github-output"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = prefix_preflight.main(
                    [
                        "--channel",
                        "egor-fedorov/safetensors-mojo",
                        "--artifact",
                        str(artifact),
                        "--github-output",
                        str(github_output),
                    ]
                )

            self.assertEqual(result, 0)
            fetch_repodata.assert_called_once_with(
                "egor-fedorov/safetensors-mojo",
                "linux-64",
            )
            self.assertEqual(
                github_output.read_text(encoding="utf-8"),
                "publish=true\n",
            )
            self.assertIn("upload is required", stdout.getvalue())

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_cli_writes_false_for_an_identical_remote_artifact(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            artifact = directory / "artifact.conda"
            artifact.write_bytes(b"artifact")
            digest = hashlib.sha256(b"artifact").hexdigest()
            fetch_repodata.return_value = {
                "packages.conda": {artifact.name: {"sha256": digest}}
            }
            github_output = directory / "github-output"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = prefix_preflight.main(
                    [
                        "--channel",
                        "channel",
                        "--subdir",
                        "noarch",
                        "--artifact",
                        str(artifact),
                        "--github-output",
                        str(github_output),
                    ]
                )

            self.assertEqual(result, 0)
            fetch_repodata.assert_called_once_with("channel", "noarch")
            self.assertEqual(
                github_output.read_text(encoding="utf-8"),
                "publish=false\n",
            )
            self.assertIn(digest, stdout.getvalue())

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_cli_fails_closed_without_writing_an_output(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        fetch_repodata.return_value = {}
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            artifact = directory / "artifact.conda"
            artifact.write_bytes(b"artifact")
            github_output = directory / "github-output"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = prefix_preflight.main(
                    [
                        "--channel",
                        "channel",
                        "--artifact",
                        str(artifact),
                        "--github-output",
                        str(github_output),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertFalse(github_output.exists())
            self.assertIn("repodata has no package index", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

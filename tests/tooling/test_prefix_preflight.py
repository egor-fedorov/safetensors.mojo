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
    def write_artifact_list(
        self,
        directory: Path,
        platforms: tuple[str, ...],
    ) -> tuple[Path, list[tuple[Path, str]]]:
        artifacts: list[tuple[Path, str]] = []
        rows: list[str] = []
        for index, platform in enumerate(platforms):
            artifact = directory / f"artifact-{index}.conda"
            artifact.write_bytes(f"artifact-{platform}".encode())
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            artifacts.append((artifact, digest))
            rows.append(
                f"{platform}\t{artifact}\t{artifact.name}\tfalse\t{digest}\n"
            )
        manifest = directory / "release-artifacts.tsv"
        manifest.write_text("".join(rows), encoding="utf-8")
        return manifest, artifacts

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

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_batch_cli_writes_only_missing_artifacts_in_input_order(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest, artifacts = self.write_artifact_list(
                directory,
                ("linux-64", "linux-aarch64", "osx-arm64"),
            )
            fetch_repodata.side_effect = [
                None,
                {
                    "packages.conda": {
                        artifacts[1][0].name: {"sha256": artifacts[1][1]}
                    }
                },
                None,
            ]
            publish_list = directory / "publish.tsv"
            github_output = directory / "github-output"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = prefix_preflight.main(
                    [
                        "--channel",
                        "egor-fedorov/safetensors-mojo",
                        "--artifact-list",
                        str(manifest),
                        "--publish-list",
                        str(publish_list),
                        "--github-output",
                        str(github_output),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args for call in fetch_repodata.call_args_list],
                [
                    ("egor-fedorov/safetensors-mojo", "linux-64"),
                    ("egor-fedorov/safetensors-mojo", "linux-aarch64"),
                    ("egor-fedorov/safetensors-mojo", "osx-arm64"),
                ],
            )
            self.assertEqual(
                github_output.read_text(encoding="utf-8"),
                f"publish_list={publish_list.resolve()}\n",
            )
            self.assertEqual(
                publish_list.read_text(encoding="utf-8"),
                f"linux-64\t{artifacts[0][0]}\n"
                f"osx-arm64\t{artifacts[2][0]}\n",
            )
            self.assertIn("selected 2 artifacts", stdout.getvalue())

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_batch_cli_writes_an_empty_plan_when_every_artifact_exists(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest, artifacts = self.write_artifact_list(
                directory,
                ("linux-64",),
            )
            fetch_repodata.return_value = {
                "packages.conda": {
                    artifacts[0][0].name: {"sha256": artifacts[0][1]}
                }
            }
            publish_list = directory / "publish.tsv"
            github_output = directory / "github-output"

            result = prefix_preflight.main(
                [
                    "--channel",
                    "channel",
                    "--artifact-list",
                    str(manifest),
                    "--publish-list",
                    str(publish_list),
                    "--github-output",
                    str(github_output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(publish_list.read_text(encoding="utf-8"), "")
            self.assertEqual(
                github_output.read_text(encoding="utf-8"),
                f"publish_list={publish_list.resolve()}\n",
            )

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_batch_cli_fails_closed_without_writing_partial_outputs(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest, artifacts = self.write_artifact_list(
                directory,
                ("linux-64", "osx-arm64"),
            )
            fetch_repodata.side_effect = [
                None,
                {
                    "packages.conda": {
                        artifacts[1][0].name: {"sha256": "0" * 64}
                    }
                },
            ]
            publish_list = directory / "publish.tsv"
            github_output = directory / "github-output"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = prefix_preflight.main(
                    [
                        "--channel",
                        "channel",
                        "--artifact-list",
                        str(manifest),
                        "--publish-list",
                        str(publish_list),
                        "--github-output",
                        str(github_output),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertFalse(publish_list.exists())
            self.assertFalse(github_output.exists())
            self.assertIn("different SHA-256", stderr.getvalue())

    @patch("tools.release.prefix_preflight.fetch_repodata")
    def test_batch_cli_rejects_changed_local_bytes_before_network(
        self,
        fetch_repodata: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest, artifacts = self.write_artifact_list(
                directory,
                ("linux-64",),
            )
            artifacts[0][0].write_bytes(b"changed")
            publish_list = directory / "publish.tsv"
            github_output = directory / "github-output"

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = prefix_preflight.main(
                    [
                        "--channel",
                        "channel",
                        "--artifact-list",
                        str(manifest),
                        "--publish-list",
                        str(publish_list),
                        "--github-output",
                        str(github_output),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertIn("SHA-256 does not match", stderr.getvalue())
            fetch_repodata.assert_not_called()
            self.assertFalse(publish_list.exists())
            self.assertFalse(github_output.exists())


if __name__ == "__main__":
    unittest.main()

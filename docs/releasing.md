# Releasing

Releases publish one immutable Conda artifact to
[`egor-fedorov/safetensors-mojo`](https://prefix.dev/channels/egor-fedorov%2Fsafetensors-mojo/packages/safetensors-mojo)
and attach the same bytes to the matching GitHub Release. The workflow uses
GitHub OIDC through Prefix.dev Repository Access; no Prefix.dev API token is
stored in GitHub.

## One-time Prefix.dev setup

Open the channel's **Settings → Repository Access**, add a GitHub repository
rule, and use these values:

- owner: `egor-fedorov`
- repository: `safetensors.mojo`
- workflow filename: `release.yml`
- access: **Read/write**
- environment: leave empty

The rule is restricted to this repository and workflow. The workflow requests
`id-token: write` only for the publishing job.

## Release checklist

1. Update both version fields in `pixi.toml`, update user-facing documentation,
   and merge the change into `main`.
2. Run `pixi run all` from a clean checkout.
3. Prepare the curated English GitHub Release body in a temporary Markdown
   file. Start it with `## Summary`; do not add a level-one heading because
   GitHub already renders the release title. Include highlights, installation,
   compatibility, deliberate limitations, and the full-changelog comparison
   link.
4. Create an annotated `vMAJOR.MINOR.PATCH` tag at the release commit using the
   complete Markdown body as its annotation, inspect it, and push it:

   ```bash
   release_tag=vMAJOR.MINOR.PATCH
   git tag --annotate --cleanup=verbatim "${release_tag}" --file "/tmp/safetensors-${release_tag}.md"
   git tag --list "${release_tag}" --format='%(contents)'
   git push origin "${release_tag}"
   ```

   `--cleanup=verbatim` is required because Git's default cleanup removes
   Markdown heading lines that begin with `#`.

5. Confirm that both the CI and Release workflows pass.
6. Confirm that the package is visible on Prefix.dev and that the matching
   GitHub Release contains the `.conda` asset and the complete curated body.

The Release workflow checks that the tag and both manifest versions agree,
runs the repository checks, builds exactly one package, installs it from a
freshly indexed local channel, and compiles and runs a Mojo consumer before it
publishes anything. It creates the GitHub Release with that exact artifact and
then uploads the same bytes to Prefix.dev. If a rerun finds an existing GitHub
package asset with the same immutable filename, those published bytes are
canonical and the fresh nondeterministic build is ignored.

When it creates a release, the workflow copies the annotated tag message into
the GitHub Release body. GitHub-generated notes are intentionally not used:
they can be useful as a commit index, but they do not communicate the public
API, installation command, compatibility boundary, or deliberate scope of a
release. The full changelog remains a link at the end of the curated body.

Release policy checks live in `tools/release/validate_release.py` and
`tools/release/prefix_preflight.py`. Their unit tests run as part of
`pixi run check`, so tag validation, Prefix.dev response handling, and
immutable-filename checks are exercised by ordinary CI rather than only during
a release.

For an existing tag created before the workflow was added, run the Release
workflow manually and supply the tag name. The tagged source remains the build
input; only the release tooling is taken from the workflow revision. The
workflow invokes release helpers from its separate `release-tooling` checkout
and runs the Mojo smoke consumer stored with the tagged source, so the consumer
matches that release's public API. The `v0.1.0` tag predates the tagged consumer
file, so that one tag uses the release tooling's minimal format-core fallback.
Any later tag without its own consumer is rejected rather than receiving the
weaker fallback.

Conda package filenames are immutable release coordinates. Never replace an
existing remote filename with different bytes. If a package must be rebuilt,
increment the package version or build number and publish a new artifact.

Published annotated tags are immutable release identities. Never move or
recreate one to correct presentation. If a published release body needs a
presentation-only correction, edit the GitHub Release body and leave the tag
object and target commit unchanged.

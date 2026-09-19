# Releasing

Releases publish one immutable Conda artifact per declared native platform to
[`egor-fedorov/safetensors-mojo`](https://prefix.dev/channels/egor-fedorov%2Fsafetensors-mojo/packages/safetensors-mojo)
and attach the same bytes to the matching GitHub Release. The current manifest
declares `linux-64`, `linux-aarch64`, and `osx-arm64`. The workflow uses GitHub
OIDC through Prefix.dev Repository Access; no Prefix.dev API token is stored
in GitHub.

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
   and merge the change into `main`. For a compiler upgrade, pin the workspace
   `mojo` dependency and the package build, host, and run `mojo-compiler`
   dependencies to the same exact version, then regenerate the lockfile for
   all declared platforms. Version 0.8.0 targets Mojo 1.1.0; version 0.7.0
   remains paired with Mojo 1.0.0.
2. Run `pixi run all` from a clean checkout. This verifies the current native
   host; the release workflow repeats the checks on every declared platform.
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
6. Confirm that every declared platform subdirectory is visible on Prefix.dev
   and that the matching GitHub Release contains all `.conda` assets and the
   complete curated body.

The Release workflow checks that the tag, both manifest versions, and declared
platform set agree, then pins both the tag object and its target commit. A
native job on each platform runs the repository checks, obtains exactly one
package, installs it from a freshly indexed local channel, and compiles and
runs a Mojo consumer. The smoke test reads the expected compiler version from
the package's source manifest and verifies the installed compiler and package
dependency against that exact pin. A single publication job proceeds only after
all native jobs succeed, validates the complete artifact set, preflights every
Prefix.dev subdirectory before the first write, creates or updates the GitHub
Release, and uploads the identical bytes to Prefix.dev.

Multi-platform package build strings begin with `linux64_`, `linuxaarch64_`,
or `osxarm64_`. Conda channels already separate packages by platform, but the
prefix also gives each attachment a unique immutable filename in one GitHub
Release. A rerun discovers an existing matching GitHub asset before building,
treats those published bytes as canonical, and smoke-tests them again on their
native runner. It builds only missing platform assets. The publisher downloads
the verified results and rejects any filename or byte change observed while the
workflow was running.

When it creates a release, the workflow copies the annotated tag message into
the GitHub Release body. GitHub-generated notes are intentionally not used:
they can be useful as a commit index, but they do not communicate the public
API, installation command, compatibility boundary, or deliberate scope of a
release. The full changelog remains a link at the end of the curated body.

The workflow keeps runner selection, permissions, checkouts, native builds, and
artifact transfer declarative. Tested Python helpers in `tools/release/` own the
procedural policy: `validate_release.py` validates the manifest and exact tag
identity, `artifacts.py` validates the canonical package set,
`github_release.py` handles idempotent GitHub asset reuse and publication, and
`prefix_preflight.py` builds the all-platform Prefix.dev upload plan. Their unit
tests run as part of `pixi run check`, so these rules are exercised by ordinary
CI rather than only during a release.

For an existing tag created before the workflow was added, run the Release
workflow manually and supply the tag name. The tagged source and its declared
platform list remain the build input; only the release tooling is taken from
the workflow revision. Single-platform historical manifests retain their
original unprefixed build string. The workflow invokes release helpers from its
separate `release-tooling` checkout and passes the tagged `pixi.toml` to the
smoke test through `--manifest`. The local default is the current project's
manifest. The workflow also runs the Mojo smoke consumer stored with the tagged
source, so both the compiler expectation and consumer match that release rather
than the current tooling checkout. The `v0.1.0` tag predates the tagged consumer
file, so that one tag uses the release tooling's minimal format-core fallback.
Any later tag without its own consumer is rejected rather than receiving the
weaker fallback.

## Community channel updates

After the GitHub Release is published, submit the matching version and release
commit to `modular/modular-community`, with build number zero for a new version.
Pin `mojo-compiler` to the exact build compiler in the recipe's build, host, and
run requirements; for 0.8.0, use `==1.1.0` in all three. A broad
`pin_compatible()` range can let the solver install a compiler that cannot
import the compiled `.mojoc` artifact.

If a previously published artifact has an overly broad compiler dependency,
request a channel metadata correction rather than rebuilding or replacing its
bytes. Users of 0.7.0 should explicitly pin `mojo-compiler ==1.0.0`.

After the recipe PR is merged, verify that the publication workflow succeeds,
that the new version appears on Prefix.dev for all three platforms, and that
the published packages pass native installation and import smoke tests. A
merged recipe alone does not confirm publication.

## Immutable release coordinates

Conda package filenames are immutable release coordinates. Never replace an
existing remote filename with different bytes. If a package must be rebuilt,
increment the package version or build number and publish a new artifact.

Published annotated tags are immutable release identities. Never move or
recreate one to correct presentation. If a published release body needs a
presentation-only correction, edit the GitHub Release body and leave the tag
object and target commit unchanged.

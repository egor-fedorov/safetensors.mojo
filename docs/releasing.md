# Releasing

Releases publish one immutable Conda artifact to
[`egor-fedorov/safetensors-mojo`](https://prefix.dev/egor-fedorov/safetensors-mojo)
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

1. Update both version fields in `pixi.toml`, update user-facing release notes,
   and merge the change into `main`.
2. Run `pixi run all` from a clean checkout.
3. Create an annotated `vMAJOR.MINOR.PATCH` tag at the release commit and push
   it.
4. Confirm that both the CI and Release workflows pass.
5. Confirm that the package is visible on Prefix.dev and that the matching
   GitHub Release contains the `.conda` asset.

The Release workflow checks that the tag and both manifest versions agree,
runs the format-core checks, builds exactly one package, installs it from a
freshly indexed local channel, and compiles and runs a Mojo consumer before it
publishes anything. It then uploads that exact artifact to Prefix.dev and the
GitHub Release.

Release policy checks live in `tools/validate_release.py` and
`tools/prefix_preflight.py`. Their unit tests run as part of `pixi run check`,
so tag validation, Prefix.dev response handling, and immutable-filename checks
are exercised by ordinary CI rather than only during a release.

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

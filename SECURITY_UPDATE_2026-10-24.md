# Security dependency update — 2026-10-24

Update: certifi 2026.10.19.

The eMudhra incident requires updating application-bundled CA stores as well as
the operating system. Frozen environments must be recreated from the reviewed
updated lock; refreshing only a global tool does not update an existing venv
or container. Rebuild affected container images with fresh base images and
package metadata before reuse. Existing experiment snapshots remain historical
evidence; do not reuse their network clients without remediation.

This change preserves model code, data, experimental results and unrelated
package versions. It does not certify historical experiments, rebuild deployed
images, or authorize a deployment.

## Evidence and verification

The previous pinned certifi wheel was inspected without executing its code.
Loading its PEM bundle into Python's TLS context included the affected G3 root
(SHA-256 `86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b`).
The corresponding absence assertion fails on the previous bundle and passes
on 2026.10.19. The replacement wheel SHA-256 was checked against PyPI metadata:
`42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b`.
The sdist hash is
`035c1e85572b141bf1c3a3eff0e33a8eb74917522f2559075ff8d2fea780a1ae`.
This verifies the trust-store change; it is not an exploitation or GPU test.

## Primary sources

- [Mozilla incident response](https://blog.mozilla.org/security/2026/10/20/protecting-firefox-users-following-the-compromise-of-emudhra-intermediate-certificates/)
- [Python security guidance](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)
- [uv security release](https://github.com/astral-sh/uv/releases/tag/0.13.4)
- [Node.js security release](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases)

See the pull request for repository-specific test results and limitations.

## Local validation

`uv run --frozen --python 3.12 pytest tests`: 18 passed, 2 failed while downloading model weights from huggingface.co due to a self-signed certificate in the TLS chain. TLS verification was not disabled. Ruff lint/format and mypy passed. `uv lock --check` passed; all non-certifi package metadata is preserved.

## Draft verification follow-up (2026-10-24)

The two feature tests were unit checks of extractor types and output keys but
implicitly downloaded pretrained weights. A network-forbidden fixture first
made both fail with an explicit assertion (2 failed / 6 passed), independently
of the external TLS error. They now load a real, locally created ResNet18
checkpoint, retaining their original output assertions. An additional test
checks that the default branch requests pretrained=True. Production feature
code is unchanged; the added default-branch test uses a model stub and is not
a pretrained-download integration test.

Local macOS: all 21 tests pass, Ruff lint/format and mypy pass. The existing
empty-input NumPy warnings are preserved. TLS verification remains enabled;
the live Hugging Face endpoint still fails certificate validation with both
certifi and the repaired system CA, so that external download is not certified.
No unverified CA was trusted and no insecure download was used.

The CI compose file overrode the Dockerfile's Ubuntu 22.04 default with 20.04.
The [deadsnakes supported distributions](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)
list 22.04/24.04; Python 3.12 is provided for 22.04. The override now uses 22.04.
The unchanged Dockerfile successfully builds a separate local review image.
The image contains pip 26.2.2 and uv 0.13.5.

Linux arm64 local container: all 21 tests pass with networking disabled and a
read-only filesystem. Ruff lint/format, mypy and mdformat checks pass. The
Docker Desktop Documents mount was unavailable, so only non-secret test/source
files were copied into a separate test image; host sharing settings were not
changed. The moved venv's Python entry point was used with `python -m mypy` and
`python -m mdformat`, avoiding its original-path script shebangs.

A separate network-enabled probe inside the review container returned PyPI
HTTP 200 and rejected all four public G3/AAA eMudhra test hosts, with and without
CT. No credentials or host directories were supplied. The local Linux arm64
result is not an executed GitHub Linux amd64 job. GPU/model-quality evaluation
and the external pretrained download remain outside this unit-test evidence.

Commands: `.venv/bin/python -m pytest tests/mensura/v_info/test_feature.py -q`
for the network-forbidden RED; `.venv/bin/python -m pytest tests -q` for GREEN.
Local CI image: `docker build --pull -f environments/Dockerfile --build-arg BASE_IMAGE=ubuntu:22.04 --build-arg RUN_UV_SYNC_AT_BUILD_TIME=true --build-arg PROJECT_NAME=ascender -t mensura-security-review:20261024 .`.
Container tests used `python -m pytest tests -q -p no:cacheprovider`,
`ruff check mensura tests scripts`, `ruff format --check mensura tests scripts`,
`python -m mypy --cache-dir /tmp/mypy mensura` and `python -m mdformat --check *.md`.
The latter commands used the image's `/home/challenger/.venv/bin/python` and Ruff.

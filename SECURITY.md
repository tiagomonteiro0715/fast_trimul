# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
suspected vulnerability.

- **Preferred:** use GitHub's private reporting — go to the repo's
  [**Security** tab → **Report a vulnerability**](https://github.com/tiagomonteiro0715/fast_trimul/security/advisories/new).
  This opens a confidential advisory only the maintainer can see.
- **Alternative:** email **monteiro.t@northeastern.edu** with `[fast_trimul security]`
  in the subject.

Please include enough to reproduce: the affected version, your environment
(GPU / CUDA driver, `torch`, `cuda-python`, `nvidia-cutlass-dsl` versions), and a
minimal example.

## What to expect

This is a small, single-maintainer project, so timelines are best-effort:

- Acknowledgement within about **5 business days**.
- If confirmed, a fix will be prepared privately and released to PyPI, with the
  advisory published (and you credited, if you'd like) once a fixed version is out.

## Supported versions

Only the **latest** release on [PyPI](https://pypi.org/project/fast_trimul/) is
supported. Please upgrade before reporting, in case the issue is already fixed.

## Scope

In scope: the `fast_trimul` package itself.

Out of scope: vulnerabilities in dependencies (`torch`, `nvidia-cutlass-dsl`,
`cuda-python`, …) — report those to the respective projects; NVIDIA CUTLASS, on
which the GEMM kernel is based; and issues that require a compromised local machine
or an already-malicious model/weights file.

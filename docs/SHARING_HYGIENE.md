# Sharing hygiene

This repo is shared with coworkers and pushed to GitHub. It also contains a
genomics-download workflow that authenticates to an external service. Rules for
what leaves the machine.

Referenced by [PROJECT_POLICY.md](PROJECT_POLICY.md) P11.

## Never commit

- **Session cookies, tokens, API keys, passwords.** Not in a file, not in a
  script literal, not in a committed shell history.
- **Credentials on a command line.** `curl --data-urlencode "password=..."`
  leaks via `ps` and shell history. Read secrets from an environment variable or
  an interactive prompt.
- **A machine-specific `config.yaml`** (P4). Commit `config.example.yaml`; keep
  real values in env vars or a gitignored local file.
- **Personal email addresses as code literals.** Use an env var
  (`JGI_USERNAME`, etc.) with a comment saying what to set.
- Raw dumps of external-service responses that might carry account or quota
  detail — check before committing; the JGI search-response shape is fine, an
  auth response is not.

Lab-internal infrastructure names (the HPC private IP `172.20.97.21`, the SMB
host `R402-NAS…local`, the `hsushen` username) are **low risk** — RFC1918, no
passwords — but prefer to centralize them in one place (a gitignored
`config.local.sh` or env) rather than scattering them across scripts.

## Pre-push scrub

Before `git push`, or before sharing a branch:

```
git diff --cached -U0 | grep -nEi 'password|secret|token|api[_-]?key|BEGIN [A-Z ]*PRIVATE KEY|jgi_session|Cookie:'
git ls-files | grep -iE 'cookie|secret|token|credential|\.env$|\.pem$|\.key$|id_rsa'
```

Any hit → stop and resolve before pushing.

## Onboarding via a chatbot

`SingleCellQuantificationHPC/README.md` currently tells a new user to paste
project facts (script names, `BASE_MOVIE_ROOT`, NAS mount details) into ChatGPT
for setup help. That is acceptable for generic install help but should not
include: credentials, the NAS/HPC hostnames or IPs, or internal file-path
layouts beyond what the person already needs. Prefer pointing coworkers at
`COWORKER_GUIDE.md` and this repo's own docs.

## Already in the repo — remediation

| Item | Risk | Action | Decision owner |
|---|---|---|---|
| `cookies` (a `.jgi.doe.gov` `jgi_session` cookie, committed since the initial commit) | Credential. Likely expired (dates to 2026-03), but grants JGI account access if live. | 1. `git rm --cached cookies`, add to `.gitignore`. 2. Log out / invalidate that JGI session at the portal. 3. Decide whether to purge from history (`git filter-repo`) — this rewrites `main` and the feature branches and everyone re-clones. | **user** — steps 2 and 3 are yours to call |
| `JGI_fungal_download_v3.py` — email literal + `password=` on the `curl` line | Email in cleartext; password visible in `ps` / shell history at runtime. | Move username to `JGI_USERNAME` env var; pass the password via `curl --config -` / `-K` from stdin or a prompt, not argv. | user (small code change, do when convenient) |
| `payload.json`, `output.json` — JGI API request/response dumps | None (IDs and search stats only). | Optional: gitignore future dumps. Leave as-is otherwise. | — |
| HPC IP / NAS host / `hsushen` in ~8 tracked scripts and docs | Low (internal, no secrets). | Optional: centralize into a gitignored `config.local.sh`. Not urgent. | — |

## Rules

- A secret that has been committed is considered compromised: remove it **and**
  rotate/revoke it at the provider. Removing the file is not enough.
- New external-service integrations read credentials from the environment, never
  from a committed file or a code literal.
- Run the pre-push scrub before sharing any branch that touches the genomics
  workflow, shell scripts, or config.

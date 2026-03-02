# Agent Notes
## Best Practices
- Run linting and fix lint errors to all code changes.

## Python and Tooling
- CI installs Hatch with `pipx install hatch`.
- Supported test matrix in CI: Python `3.8` through `3.13` (`hatch test -py <version>`).

## Common Commands
- Run tests (default): `hatch run test`
- Run tests for a specific Python version: `hatch test -py 3.12`
- Run coverage: `hatch run cov`
- Run Ruff locally: `ruff check .`
- Run mypy via Hatch types env: `hatch run types:check`
- Build package: `hatch build`

## CI Workflows
- `.github/workflows/linters-and-test.yml`
  - Lint job uses `chartboost/ruff-action@v1`.
  - Test job runs `hatch test -py <matrix-version>`.
- `.github/workflows/python-publish.yml`
  - On release publish, runs `hatch build` then publishes via `pypa/gh-action-pypi-publish@release/v1`.
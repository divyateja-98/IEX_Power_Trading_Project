# CI/CD

CI/CD should validate code quality, artifact availability, and reproducibility.

## Suggested Gates

1. Install dependencies.
2. Run formatting and lint checks.
3. Run smoke validation.
4. Run unit tests.
5. Run `dvc status` and `dvc repro` in controlled build environments.
6. Build backend and frontend images.
7. Deploy to Kubernetes after approval.

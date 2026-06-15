# MLOps Architecture

This directory contains the enterprise MLOps control plane for the IEX MCP
forecasting project.

| Area | Purpose |
| --- | --- |
| `dvc/` | Data and model versioning guidance around the root `dvc.yaml` pipeline |
| `validation/` | Dataset, feature, model, and business-rule validation assets |
| `feature_store/` | Feature definitions and registry metadata |
| `experiments/` | Experiment tracking conventions and reproducibility notes |
| `lineage/` | Dataset, model, and report lineage records |
| `monitoring/` | Data drift, model quality, and service health monitoring |
| `deployment/` | Service packaging and runtime configuration |
| `ci_cd/` | Continuous integration and delivery workflow definitions |
| `kubernetes/` | Kubernetes manifests for backend and frontend workloads |

The existing scripts remain the system of record for current behavior. New MLOps
assets document and orchestrate those scripts without changing the forecasting
logic.

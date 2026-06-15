# Kubernetes

Kubernetes manifests for serving the backend API and Streamlit frontend.

Apply after building and publishing container images:

```powershell
kubectl apply -f mlops/kubernetes/backend.yaml
kubectl apply -f mlops/kubernetes/frontend.yaml
```

Update image names before production deployment.

# GCP Workload Identity Federation IAM Guide (Zero Context)

## Why this document exists

This guide explains, from first principles, why a GitHub Actions deployment to Google Cloud can fail with IAM errors, and exactly how to fix it.

It assumes you are new to:
- Google Cloud IAM
- Service accounts
- Workload Identity Federation (WIF)
- GitHub OIDC authentication

It also explains the companion script:
- `infra/gcp/scripts/fix_wif_iam_permissions.sh`

---

## 1. The deployment model in this repository

Your workflow (`.github/workflows/deploy-staging-gcp.yml`) does this:

1. GitHub Actions starts a job.
2. The job authenticates to GCP using `google-github-actions/auth@v3`.
3. That authentication uses Workload Identity Federation (OIDC), not a static JSON key.
4. The workflow then runs Terraform to create/update GCP resources.

This model is secure and modern, but IAM permissions must be configured correctly at two levels.

---

## 2. The two IAM permission layers you must configure

### Layer A: Permission for GitHub to impersonate your service account

GitHub's OIDC principal must be allowed to impersonate the target service account.

That requires service-account-level IAM bindings on the service account (for the `principalSet://...` identity):

- `roles/iam.workloadIdentityUser`
- `roles/iam.serviceAccountTokenCreator`

If missing, you will see errors like:

- `Permission 'iam.serviceAccounts.getAccessToken' denied`

### Layer B: Permission for the service account to manage resources

After impersonation succeeds, Terraform runs as the service account.
The service account itself needs project-level roles for the resources you manage.

For this repo's deployment workflow, minimum roles are:

- `roles/serviceusage.serviceUsageAdmin` (for `google_project_service` resources)
- `roles/artifactregistry.admin` (mirror/push deploy images into Artifact Registry)
- `roles/run.admin` (Cloud Run)
- `roles/compute.admin` (GPU Compute Engine path)
- `roles/iam.serviceAccountUser` (commonly required for service operations)

---

## 3. Key terminology (quick definitions)

### Service account
A non-human identity in GCP that automation uses.

Example in this repo:
- `github-actions@miniflow-489011.iam.gserviceaccount.com`

### Workload Identity Pool
A trust boundary in GCP that accepts identities from an external issuer (GitHub OIDC).

Example:
- `github-pool`

### Workload Identity Provider
The specific OIDC provider config inside the pool.

Example:
- `github-provider`

### principalSet
The exact external identity pattern that receives IAM binding.
In this repo, it is bound to `attribute.repository=KaranjotVendal/MiniFlow`.

---

## 4. Script: what it fixes

Script path:
- `infra/gcp/scripts/fix_wif_iam_permissions.sh`

### What it validates first

- `gcloud` is installed
- project exists
- service account exists
- workload identity pool exists
- workload identity provider exists

### What it grants

#### On service account (for GitHub principalSet)

- `roles/iam.workloadIdentityUser`
- `roles/iam.serviceAccountTokenCreator`

#### On project (for service account)

- `roles/serviceusage.serviceUsageAdmin`
- `roles/artifactregistry.admin`
- `roles/run.admin`
- `roles/compute.admin`
- `roles/iam.serviceAccountUser`

The script is idempotent: running it multiple times is safe.

---

## 5. How to run the script

From repository root:

```bash
bash infra/gcp/scripts/fix_wif_iam_permissions.sh \
  --project-id miniflow-489011 \
  --repo KaranjotVendal/MiniFlow
```

Optional parameters:

```bash
bash infra/gcp/scripts/fix_wif_iam_permissions.sh \
  --project-id miniflow-489011 \
  --repo KaranjotVendal/MiniFlow \
  --service-account github-actions@miniflow-489011.iam.gserviceaccount.com \
  --pool github-pool \
  --provider github-provider
```

---

## 6. What successful output should look like

You should see log lines such as:

- `Granting service-account-level role roles/iam.workloadIdentityUser ...`
- `Granting service-account-level role roles/iam.serviceAccountTokenCreator ...`
- `Granting project-level role roles/serviceusage.serviceUsageAdmin ...`
- `Granting project-level role roles/artifactregistry.admin ...`
- `Granting project-level role roles/run.admin ...`
- `Granting project-level role roles/compute.admin ...`

At the end, the script prints expected GitHub secret values.

---

## 7. Post-fix verification checklist

### Verify service-account IAM bindings

```bash
gcloud iam service-accounts get-iam-policy \
  github-actions@miniflow-489011.iam.gserviceaccount.com \
  --project=miniflow-489011
```

You should see `bindings:` entries containing:

- `roles/iam.workloadIdentityUser`
- `roles/iam.serviceAccountTokenCreator`

and member resembling:

- `principalSet://iam.googleapis.com/projects/<project-number>/locations/global/workloadIdentityPools/github-pool/attribute.repository/KaranjotVendal/MiniFlow`

### Verify project-level roles for service account

```bash
gcloud projects get-iam-policy miniflow-489011 \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:github-actions@miniflow-489011.iam.gserviceaccount.com" \
  --format="table(bindings.role, bindings.members)"
```

You should see at least:

- `roles/serviceusage.serviceUsageAdmin`
- `roles/artifactregistry.admin`
- `roles/run.admin`
- `roles/compute.admin`
- `roles/iam.serviceAccountUser`

### Verify GitHub environment secrets

Because workflow jobs use `environment: miniflow`, ensure those secrets exist in that environment:

- `GCP_PROJECT_ID`
- `GCP_SERVICE_ACCOUNT`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`

---

## 8. Common errors and exact causes

### Error: `iam.serviceAccounts.getAccessToken denied`

Cause:
- Missing `roles/iam.workloadIdentityUser` and/or `roles/iam.serviceAccountTokenCreator` on service account for GitHub principalSet.

Fix:
- Run script in this guide.

### Error: `workflow auth inputs empty`

Cause:
- Secrets exist at environment scope but job is not bound to that environment, or secret name mismatch.

Fix:
- Ensure workflow jobs set `environment: miniflow`.
- Verify secret names exactly match workflow references.

### Error enabling APIs in Terraform

Cause:
- Service account missing `roles/serviceusage.serviceUsageAdmin`.

Fix:
- Run script in this guide.

---

## 9. Security notes

- This setup uses short-lived OIDC tokens, not long-lived JSON keys.
- Scope is limited to one repository by binding principalSet to `attribute.repository`.
- Prefer least privilege for long-term hardening; roles above are practical for getting deployment unblocked.

---

## 10. Minimal runbook

1. Run script:
   ```bash
   bash infra/gcp/scripts/fix_wif_iam_permissions.sh --project-id miniflow-489011 --repo KaranjotVendal/MiniFlow
   ```
2. Confirm service-account policy includes required bindings.
3. Confirm GitHub environment secrets are present in `miniflow` environment.
4. Re-run workflow:
   - `Deploy Staging (GCP)`
   - `deployment_type=cpu`
   - image tag set to a known existing GHCR tag.

---

## 11. Related files in this repository

- Workflow: `.github/workflows/deploy-staging-gcp.yml`
- Terraform root: `infra/gcp/terraform/`
- IAM fix script: `infra/gcp/scripts/fix_wif_iam_permissions.sh`

---

## 12. Optional cleanup after successful deploy

If you want stricter least-privilege later, reduce project roles once stable behavior is confirmed. Keep `workloadIdentityUser` and `serviceAccountTokenCreator` unless you change auth model.

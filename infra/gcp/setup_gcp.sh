#!/bin/bash

   PROJECT_ID="miniflow-489011"
   REPO="KaranjotVendal/MiniFlow"

   echo "=== Setting up GCP for MiniFlow ==="

   # Enable APIs
   echo "Enabling APIs..."
   gcloud services enable run.googleapis.com --project=$PROJECT_ID
   gcloud services enable logging.googleapis.com --project=$PROJECT_ID
   gcloud services enable iamcredentials.googleapis.com --project=$PROJECT_ID

   # Create service account
   echo "Creating service account..."
   gcloud iam service-accounts create github-actions \
     --display-name="GitHub Actions" \
     --project=$PROJECT_ID || echo "Service account may already exist"

   SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

   # Grant permissions
   echo "Granting permissions..."
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:$SERVICE_ACCOUNT" \
     --role="roles/run.admin" || echo "Permission may already exist"

   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:$SERVICE_ACCOUNT" \
     --role="roles/iam.serviceAccountUser" || echo "Permission may already exist"

   # Create Workload Identity Pool
   echo "Creating Workload Identity Pool..."
   gcloud iam workload-identity-pools create github-pool \
     --location="global" \
     --display-name="GitHub Actions Pool" \
     --project=$PROJECT_ID || echo "Pool may already exist"

   # Create Provider
   echo "Creating Workload Identity Provider..."
   gcloud iam workload-identity-pools providers create-oidc github-provider \
     --location="global" \
     --workload-identity-pool="github-pool" \
     --display-name="GitHub Actions Provider" \
     --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
     --attribute-condition="assertion.repository=='${REPO}'" \
     --issuer-uri="https://token.actions.githubusercontent.com" \
     --project=$PROJECT_ID || echo "Provider may already exist"

   # Get project number
   PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

   # Allow GitHub to impersonate service account
   echo "Configuring service account IAM..."
   gcloud iam service-accounts add-iam-policy-binding \
     $SERVICE_ACCOUNT \
     --role="roles/iam.workloadIdentityUser" \

 --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attri
 bute.repository/${REPO}" \
     --project=$PROJECT_ID || echo "IAM binding may already exist"

   # Output the provider resource name
   echo ""
   echo "=== Setup Complete ==="
   echo "Add these GitHub Secrets:"
   echo ""
   echo "GCP_PROJECT_ID: ${PROJECT_ID}"
   echo "GCP_SERVICE_ACCOUNT: ${SERVICE_ACCOUNT}"
   echo "GCP_WORKLOAD_IDENTITY_PROVIDER:
 projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"

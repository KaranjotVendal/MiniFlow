#!/bin/bash
# find-gpu-zone.sh - Find available GPU zone in GCP
#
# Usage:
#   ./find-gpu-zone.sh [PROJECT_ID] [MACHINE_TYPE] [GPU_TYPE] [GPU_COUNT]
#
# Examples:
#   # Find P100 (default)
#   ./find-gpu-zone.sh
#
#   # Find T4
#   ./find-gpu-zone.sh miniflow-489011 n1-standard-4 nvidia-tesla-t4 1
#
#   # Find V100
#   ./find-gpu-zone.sh miniflow-489011 n1-standard-8 nvidia-tesla-v100 1
#
# Supported GPU types:
#   - nvidia-tesla-p100  (16GB, most available)
#   - nvidia-tesla-t4    (16GB, good for inference)
#   - nvidia-tesla-v100  (16GB or 32GB, powerful)
#   - nvidia-tesla-k80   (12GB per GPU, older)
#   - nvidia-tesla-a100  (40GB or 80GB, expensive)
#
# Note: P100 is recommended for MiniFlow as it offers good performance
# with better availability and lower cost than newer GPUs.

set -e

# Configuration
PROJECT_ID="${1:-miniflow-489011}"
MACHINE_TYPE="${2:-n1-standard-4}"
GPU_TYPE="${3:-nvidia-tesla-p100}"
GPU_COUNT="${4:-1}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==================================="
echo "GCP GPU Zone Finder"
echo "==================================="
echo ""
echo "Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: $GPU_TYPE (x$GPU_COUNT)"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: PROJECT_ID not set${NC}"
    echo "Usage: $0 [PROJECT_ID] [MACHINE_TYPE] [GPU_TYPE] [GPU_COUNT]"
    echo "Example: $0 miniflow-489011 n1-standard-4 nvidia-tesla-t4 1"
    exit 1
fi

# Zones to try in order of preference
# Based on observed availability patterns
ZONES=(
    "us-central1-f"     # Often has spare capacity
    "us-central1-b"     # Good availability
    "us-central1-c"     # Alternative
    "us-central1-a"     # Usually busy but worth trying
    "us-west1-b"        # Different region
    "us-west1-c"        # Different region
    "us-east1-b"        # East coast option
    "us-east1-c"        # East coast option
    "europe-west4-a"    # Europe option
    "asia-east1-a"      # Asia option
)

echo "Testing GPU availability in ${#ZONES[@]} zones..."
echo "This may take a minute or two..."
echo ""

delete_test_instance() {
    local zone="$1"
    for attempt in {1..5}; do
        if gcloud compute instances delete gpu-test-$$ \
            --zone="$zone" \
            --project="$PROJECT_ID" \
            --quiet 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    return 1
}

# Test each zone
for zone in "${ZONES[@]}"; do
    echo -n "Testing $zone... "

    # Try to create a test instance
    # Use preemptible for faster/cheaper test
    # Use --async with timeout to avoid hanging
    # Capture both stdout and stderr without exiting on non-zero status
    CREATE_RESULT=0
    set +e
    CREATE_OUTPUT=$(gcloud compute instances create gpu-test-$$ \
        --zone="$zone" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --maintenance-policy=TERMINATE \
        --preemptible \
        --project="$PROJECT_ID" \
        --quiet \
        --async \
        --format="none" 2>&1)
    CREATE_RESULT=$?
    set -e

    # Check if create command itself failed (non-zero exit)
    if [ $CREATE_RESULT -ne 0 ]; then
        # Creation command failed - check error type
        if echo "$CREATE_OUTPUT" | grep -q "ZONE_RESOURCE_POOL_EXHAUSTED"; then
            echo -e "${RED}EXHAUSTED${NC}"
        elif echo "$CREATE_OUTPUT" | grep -q "QUOTA_EXCEEDED"; then
            echo -e "${YELLOW} QUOTA EXCEEDED${NC}"
            echo -e "${YELLOW}  Request quota: https://console.cloud.google.com/iam-admin/quotas${NC}"
            exit 1
        elif echo "$CREATE_OUTPUT" | grep -q "not supported"; then
            echo -e "${YELLOW} NOT SUPPORTED${NC}"
        else
            echo -e "${RED} FAILED${NC}"
        fi
        # Continue to next zone
        continue
    fi

    # Creation command succeeded - wait for instance to provision
    # Poll for RUNNING status (instance may take time to provision)
    echo -n "Waiting for instance... "
    for i in {1..12}; do
        sleep 10
        INSTANCE_STATUS=$(gcloud compute instances describe gpu-test-$$ \
            --zone="$zone" \
            --project="$PROJECT_ID" \
            --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
        if [ "$INSTANCE_STATUS" = "RUNNING" ]; then
            break
        fi
        echo -n "."
    done
    echo ""

    if [ "$INSTANCE_STATUS" = "RUNNING" ]; then
        # Success! Delete the test instance
        if ! delete_test_instance "$zone"; then
            echo "Warning: could not delete test instance in $zone after retries"
        fi

        echo -e "${GREEN}AVAILABLE${NC}"
        echo ""
        echo -e "${GREEN}===================================${NC}"
        echo -e "${GREEN}FOUND AVAILABLE ZONE${NC}"
        echo -e "${GREEN}===================================${NC}"
        echo ""
        echo "Zone: $zone"
        echo ""
        echo "Deploy with Terraform:"
        echo "  terraform apply -var='gpu_zone=$zone'"
        echo ""
        echo "Or deploy with gcloud:"
        echo "  gcloud compute instances create miniflow-gpu \\"
        echo "    --zone=$zone \\"
        echo "    --machine-type=$MACHINE_TYPE \\"
        echo "    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \\"
        echo "    --maintenance-policy=TERMINATE \\"
        echo "    --preemptible"
        echo ""

        # Export for use in other scripts
        echo "export GPU_ZONE=\"$zone\""

        exit 0
    else
        # Instance didn't reach RUNNING state
        echo -e "${RED} FAILED TO PROVISION${NC}"
        # Clean up if partially created
        if ! delete_test_instance "$zone"; then
            echo "Warning: could not delete failed test instance in $zone after retries"
        fi
    fi
done

echo ""
echo -e "${RED}===================================${NC}"
echo -e "${RED}NO AVAILABLE ZONE FOUND${NC}"
echo -e "${RED}===================================${NC}"
echo ""
echo "All tested zones are exhausted or don't support this GPU."
echo ""
echo "Options:"
echo "1. Wait 30-60 minutes and try again (capacity changes)"
echo "2. Try a different GPU type (e.g., T4, V100, A100)"
echo "3. Deploy to Cloud Run (CPU) instead"
echo "4. Request quota increase:"
echo "   https://console.cloud.google.com/iam-admin/quotas"
echo ""
echo "Alternative: Deploy CPU version to Cloud Run:"
echo "  terraform apply -target=google_cloud_run_service.miniflow"
echo ""

exit 1

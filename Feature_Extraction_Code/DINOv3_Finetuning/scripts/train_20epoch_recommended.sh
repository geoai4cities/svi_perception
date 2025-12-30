#!/bin/bash

# 20-Epoch Recommended Training with Optimized Settings
# =====================================================
# Before running, update the paths below to match your setup

echo "========================================="
echo "DINOv3 20-Epoch Finetuning (Recommended)"
echo "========================================="

# Activate environment - UPDATE THIS PATH
# source /path/to/your/dinov3_env/bin/activate
# Or: conda activate dinov3_env

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =====================================================
# UPDATE THESE PATHS FOR YOUR SETUP
# =====================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_DIR="$(dirname "$SCRIPT_DIR")"

# Model weights - download from Meta's DINOv3 repo
WEIGHTS_DIR="${WEIGHTS_DIR:-./weights/dinov3}"
BACKBONE_WEIGHTS="${WEIGHTS_DIR}/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth"
ADE20K_WEIGHTS="${WEIGHTS_DIR}/adapters/segmentation/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"

# Dataset path - UPDATE THIS
DATA_DIR="${DATA_DIR:-./data/segmentation_dataset}"
# =====================================================

# Training configuration - 20 EPOCHS (Recommended)
EPOCHS=20
BATCH_SIZE=1
LR=1e-4
IMG_SIZE=512  # Optimized for A100 40GB memory
NUM_WORKERS=2  # Reduced workers

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${CLEAN_DIR}/checkpoints/train_20epoch_recommended_${TIMESTAMP}"

echo "Training Configuration (20 Epochs - Recommended):"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Image Size: ${IMG_SIZE}x${IMG_SIZE}"
echo "  Learning Rate: ${LR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

echo "Starting 20-epoch training..."
echo "Estimated time: ~16 hours"
echo "========================================="

# Run training with gradient-friendly proper method
cd "$CLEAN_DIR"
python scripts/train_dinov3_idd_proper.py \
    --backbone-weights "$BACKBONE_WEIGHTS" \
    --ade20k-weights "$ADE20K_WEIGHTS" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --img-size $IMG_SIZE \
    --num-workers $NUM_WORKERS \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# Check exit status
if [ $? -eq 0 ]; then
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    
    echo ""
    echo "========================================="
    echo "✅ 20-Epoch Training Completed!"
    echo "⏱️  Duration: ${HOURS}h ${MINUTES}m"
    echo "📁 Output: $OUTPUT_DIR"
    echo ""
    
    # Extract final metrics
    echo "📊 Final Training Metrics:"
    grep "Best IoU" "${OUTPUT_DIR}/training.log" | tail -1
    
    echo ""
    echo "📊 View detailed metrics:"
    echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard"
    echo ""
    echo "🎯 Model ready for inference!"
    echo "  Best model: $OUTPUT_DIR/best_model.pth"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "❌ 20-Epoch training failed!"
    echo "📄 Check log: ${OUTPUT_DIR}/training.log"
    echo "========================================="
    exit 1
fi

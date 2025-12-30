# Segmentation Feature Extraction v2.0

This directory contains scripts for extracting segmentation features from images using finetuned IDD models, with automatic saving and resume capabilities.

## Features

- **Auto-save**: Saves features every 100 images to prevent data loss
- **Resume capability**: Can resume from where it left off if interrupted
- **Background processing**: Can run in background with monitoring
- **Progress tracking**: Real-time progress monitoring and completion estimates
- **26 IDD classes**: Processes all 26 IDD segmentation classes
- **No test mode**: Directly processes all images for production use

## Files

- `extract_finetuned_features.py` - Main segmentation feature extraction script
- `run_segmentation_extraction.sh` - Shell script for background execution
- `monitor_segmentation_progress.py` - Python-based progress monitor
- `README.md` - This documentation file

## Quick Start

### 1. Start Full Extraction in Background

```bash
cd segmentation_analysis
./run_segmentation_extraction.sh start
```

This will start processing ALL images (~111,000) with resume capability.

### 2. Monitor Progress

```bash
# Option 1: Using shell script
./run_segmentation_extraction.sh status

# Option 2: Using Python monitor (more detailed)
python3 monitor_segmentation_progress.py
```

## Shell Script Commands

The `run_segmentation_extraction.sh` script provides several commands:

```bash
./run_segmentation_extraction.sh start      # Start extraction in background (all images)
./run_segmentation_extraction.sh stop       # Stop running extraction
./run_segmentation_extraction.sh restart    # Restart extraction
./run_segmentation_extraction.sh status     # Show current status
./run_segmentation_extraction.sh monitor    # Real-time log monitoring
./run_segmentation_extraction.sh logs       # Show recent logs
./run_segmentation_extraction.sh help       # Show help
```

## Progress Monitoring

### Real-time Monitoring

```bash
# Start monitoring (updates every 5 seconds)
python3 monitor_segmentation_progress.py
```

The monitor shows:
- Process status (running/stopped)
- Current progress (images processed)
- File size and age information
- Estimated completion time
- Time per image processing rate

### Status Check

```bash
./run_segmentation_extraction.sh status
```

Shows:
- Process information (PID, CPU, memory usage)
- Recent log output
- Progress file contents
- Temporary features file statistics

## Resume Capability

The script automatically detects if it can resume from a previous run:

1. **Progress file**: `../output/progress_finetuned.json`
2. **Temporary features**: `../output/temp_finetuned_segmentation_features.csv`

When resuming:
- The script loads existing progress
- Continues from the last processed image
- Validates data integrity
- Automatically resumes where it left off

## Auto-save Features

- **Progress**: Saved every 1000 images (configurable)
- **Features**: Saved every 100 images (configurable)
- **Temporary files**: Automatically cleaned up on completion

## Segmentation Classes

The script processes all 26 IDD classes:
- Road, sidewalk, building, wall, fence, pole, traffic light, traffic sign
- Vegetation, terrain, sky, person, rider, car, truck, bus, train
- Motorcycle, bicycle, and more...

## Output Format

The final CSV contains:
- `image_id` - Unique image identifier
- `seg_vit_*` - Segmentation area ratios for 26 classes (26 columns)
- `beautiful`, `lively`, `boring`, `safe` - Perception scores (4 columns)
- Total: 31 columns

## Configuration

All paths and settings are configured in `../config.py`:

- Model weights and checkpoints
- Output directories
- Image manifests
- Perception score paths

## Troubleshooting

### Process Not Found

```bash
# Check if process is running
./run_segmentation_extraction.sh status

# If PID file is stale, clean it up
rm segmentation_extraction.pid
```

### Corrupted Progress

```bash
# Remove old progress files to start fresh
rm ../output/progress_finetuned.json
rm ../output/temp_finetuned_segmentation_features.csv
```

### Monitor Issues

```bash
# Check if output directory exists
ls -la ../output/

# Verify config paths
python3 -c "from config import PATHS; print(PATHS)"
```

## File Structure

```
segmentation_analysis/
├── extract_finetuned_features.py           # Main script
├── run_segmentation_extraction.sh          # Shell runner
├── monitor_segmentation_progress.py        # Progress monitor
├── logs/                                   # Log files (auto-created)
│   └── segmentation_extraction.log
├── segmentation_extraction.pid             # Process ID file
└── README.md                               # This file

../output/                                  # Output directory
├── progress_finetuned.json                 # Progress tracking
├── temp_finetuned_segmentation_features.csv # Temporary features
└── finetuned_segmentation_features.csv     # Final results
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Memory**: Monitor memory usage during long runs
3. **Storage**: Ensure sufficient disk space for temporary files
4. **Network**: For remote execution, use `screen` or `tmux`

## Example Workflow

```bash
# 1. Start extraction
./run_segmentation_extraction.sh start

# 2. Monitor progress (in another terminal)
python3 monitor_segmentation_progress.py

# 3. Check status occasionally
./run_segmentation_extraction.sh status

# 4. Stop if needed
./run_segmentation_extraction.sh stop

# 5. Resume later
./run_segmentation_extraction.sh start  # Will automatically resume
```

## Safety Features

- **Graceful shutdown**: Sends SIGTERM before SIGKILL
- **Data validation**: Checks progress vs. results consistency
- **Automatic cleanup**: Removes temporary files on completion
- **Error handling**: Continues processing on individual image failures
- **Progress persistence**: Saves state regularly to prevent data loss

## Key Differences from Object Detection

- **No test mode**: Directly processes all images
- **26 classes**: IDD segmentation classes instead of COCO detection
- **Area ratios**: Outputs percentage coverage instead of object counts
- **Specialized model**: Uses finetuned IDD segmentation model
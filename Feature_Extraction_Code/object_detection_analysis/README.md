# Object Detection Feature Extraction v2.1

This directory contains scripts for extracting object detection features from images using DINOv3, with automatic saving and resume capabilities.

## Features

- **Auto-save**: Saves features every 100 images to prevent data loss
- **Resume capability**: Can resume from where it left off if interrupted
- **Background processing**: Can run in background with monitoring
- **Progress tracking**: Real-time progress monitoring and completion estimates
- **Config-based**: Uses centralized configuration for easy setup

## Files

- `extract_detection_features_v2.py` - Main feature extraction script
- `run_feature_extraction.sh` - Shell script for background execution
- `monitor_progress.py` - Python-based progress monitor
- `README.md` - This documentation file

## Quick Start

### 1. Run Test Mode First

```bash
cd object_detection_analysis
python3 extract_detection_features_v2.py
```

This will run a test on a small subset of images to verify everything works.

### 2. Start Full Extraction in Background

```bash
# Start the extraction
./run_feature_extraction.sh start

# Check status
./run_feature_extraction.sh status

# Monitor in real-time
./run_feature_extraction.sh monitor
```

### 3. Monitor Progress

```bash
# Option 1: Using shell script
./run_feature_extraction.sh status

# Option 2: Using Python monitor (more detailed)
python3 monitor_progress.py
```

## Shell Script Commands

The `run_feature_extraction.sh` script provides several commands:

```bash
./run_feature_extraction.sh start      # Start FULL extraction in background (all images)
./run_feature_extraction.sh test       # Start TEST extraction in background (few images)
./run_feature_extraction.sh stop       # Stop running extraction
./run_feature_extraction.sh restart    # Restart extraction (auto-detects mode)
./run_feature_extraction.sh status     # Show current status
./run_feature_extraction.sh monitor    # Real-time log monitoring
./run_feature_extraction.sh logs       # Show recent logs
./run_feature_extraction.sh help       # Show help
```

**Mode Differences:**
- **`start`**: Processes ALL images with resume capability
- **`test`**: Processes few images for testing (no resume needed)
- **`restart`**: Automatically detects previous mode and restarts accordingly

## Progress Monitoring

### Real-time Monitoring

```bash
# Start monitoring (updates every 5 seconds)
python3 monitor_progress.py
```

The monitor shows:
- Process status (running/stopped)
- Current progress (images processed)
- File size and age information
- Estimated completion time
- Time per image processing rate

### Status Check

```bash
./run_feature_extraction.sh status
```

Shows:
- Process information (PID, CPU, memory usage)
- Recent log output
- Progress file contents
- Temporary features file statistics

## Resume Capability

The script automatically detects if it can resume from a previous run:

1. **Progress file**: `../output/progress_v2.json`
2. **Temporary features**: `../output/temp_detection_features_v2.csv`

When resuming:
- The script loads existing progress
- Continues from the last processed image
- Validates data integrity
- Provides option to start fresh if needed

## Auto-save Features

- **Progress**: Saved every 1000 images (configurable)
- **Features**: Saved every 100 images (configurable)
- **Temporary files**: Automatically cleaned up on completion

## Configuration

All paths and settings are configured in `../config.py`:

- Image manifests
- Output directories
- Model weights
- Target classes
- Confidence thresholds

## Troubleshooting

### Process Not Found

```bash
# Check if process is running
./run_feature_extraction.sh status

# If PID file is stale, clean it up
rm feature_extraction.pid
```

### Corrupted Progress

```bash
# Start fresh (removes old progress)
./run_feature_extraction.sh start
# Choose 'fresh' when prompted
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
object_detection_analysis/
├── extract_detection_features_v2.py  # Main script
├── run_feature_extraction.sh         # Shell runner
├── monitor_progress.py               # Progress monitor
├── logs/                            # Log files (auto-created)
│   └── feature_extraction.log
├── feature_extraction.pid           # Process ID file
└── README.md                        # This file

../output/                           # Output directory
├── progress_v2.json                 # Progress tracking
├── temp_detection_features_v2.csv   # Temporary features
└── detection_features_v2.csv        # Final results
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Memory**: Monitor memory usage during long runs
3. **Storage**: Ensure sufficient disk space for temporary files
4. **Network**: For remote execution, use `screen` or `tmux`

## Example Workflow

```bash
# 1. Start full extraction (all images)
./run_feature_extraction.sh start

# 2. Monitor progress (in another terminal)
python3 monitor_progress.py

# 3. Check status occasionally
./run_feature_extraction.sh status

# 4. Stop if needed
./run_feature_extraction.sh stop

# 5. Resume later
./run_feature_extraction.sh start  # Will automatically resume

# Alternative: Test mode for quick verification
./run_feature_extraction.sh test   # Process few images first
```

## Safety Features

- **Graceful shutdown**: Sends SIGTERM before SIGKILL
- **Data validation**: Checks progress vs. results consistency
- **Automatic cleanup**: Removes temporary files on completion
- **Error handling**: Continues processing on individual image failures
- **Progress persistence**: Saves state regularly to prevent data loss

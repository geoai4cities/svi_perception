# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment (First Time Only)
```bash
cd perception_prediction_gitrepo
source setup_experiment.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Verify data and scripts

### 2. Run Quick Test (2-3 minutes)
```bash
./run_experiment.sh --test
```

This runs a minimal experiment to verify everything works.

### 3. Run Full Experiment (2-6 hours)
```bash
# Run in foreground
./run_experiment.sh --full

# OR run in background (recommended)
./run_experiment.sh --full --background
```

### 4. Monitor Progress
```bash
# Real-time monitoring
./monitor_experiment.sh

# Or check status
./run_experiment.sh --status
```

### 5. View Results
```bash
cd experiments/*/03_results/
ls visualizations/  # View generated figures
cat metrics/all_results.csv  # View metrics
cat ../experiment_summary.json  # View summary
```

## 📊 What You Get

After running the full experiment, you'll have:
- **112 trained models** (4 perceptions × 7 deltas × 4 models)
- **12+ publication-ready figures** (PNG, PDF, SVG formats)
- **Comprehensive metrics** (F1, Accuracy, ROC-AUC, PR-AUC)
- **Detailed reports** and analysis

## 🎯 Common Commands

```bash
# Test with specific city
./run_experiment.sh --test --test-cities Mumbai

# Full experiment with background execution
./run_experiment.sh --full --background

# CPU-only mode (no GPU)
./run_experiment.sh --full --cpu-only

# Check running experiments
./run_experiment.sh --status

# Stop all experiments
./run_experiment.sh --stop

# Get help
./run_experiment.sh --help
```

## 📁 Repository Structure

```
perception_prediction_gitrepo/
├── README.md                    # Comprehensive documentation
├── QUICK_START.md               # This file
├── setup_experiment.sh          # Environment setup
├── run_experiment.sh            # Main experiment runner
├── monitor_experiment.sh        # Progress monitoring
├── requirements.txt             # Python dependencies
│
├── Input_Data/                  # Example data
│   └── dinov3_all_classes/
│       ├── beautiful_input.xlsx
│       ├── lively_input.xlsx
│       ├── boring_input.xlsx
│       └── safe_input.xlsx
│
├── core_scripts/                # Core Python modules
├── Feature_Importance/          # Feature importance analysis
├── config/                      # Configuration files
└── experiments/                 # Results (created during runs)
```

## 🔧 Troubleshooting

**Problem: Permission denied**
```bash
chmod +x *.sh
```

**Problem: Virtual environment not found**
```bash
source setup_experiment.sh
```

**Problem: Import errors**
```bash
source perception_env/bin/activate
pip install -r requirements.txt
```

**Problem: Out of memory**
- Close unnecessary applications
- Run experiments overnight
- Use `--cpu-only` flag

## 📚 Next Steps

1. **Read the full documentation**: See `README.md` for detailed information
2. **Customize experiments**: Edit `run_experiment.sh` to change parameters
3. **Run feature importance**: See `Feature_Importance/feature_importance.md`
4. **Use your own data**: Place Excel files in `Input_Data/your_dataset/`

## 💡 Pro Tips

- Use `--background` flag for long experiments
- Monitor progress with `./monitor_experiment.sh`
- Results are timestamped and saved automatically
- Check logs in `experiments/*/05_logs/*.log`

## 🆘 Need Help?

- View help: `./run_experiment.sh --help`
- Read full docs: `README.md`
- Check logs: `experiments/*/05_logs/*.log`
- Open an issue on GitHub

---

**Happy Experimenting!** 🎯📊

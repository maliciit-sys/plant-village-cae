#!/bin/bash
# Save this as commit_all.sh and run with: bash commit_all.sh

# ============================================
# PHASE 1: PROJECT SETUP (Commits 1-5)
# ============================================

# Commit 1: Initial project structure
git add .gitignore
git commit -m "chore: add .gitignore for Python and data files"

# Commit 2: Add license
git add LICENSE
git commit -m "docs: add MIT license"

# Commit 3: Add requirements
git add requirements.txt
git commit -m "chore: add requirements.txt with dependencies"

# Commit 4: Create directory structure
git add data/.gitkeep data/raw/.gitkeep data/processed/.gitkeep
git commit -m "chore: create data directory structure"

# Commit 5: Add empty directories
git add checkpoints/.gitkeep logs/.gitkeep
git commit -m "chore: add checkpoints and logs directories"

# ============================================
# PHASE 2: CONFIGURATION FILES (Commits 6-10)
# ============================================

# Commit 6: Dataset config
git add config/dataset_config.json
git commit -m "config: add dataset configuration"

# Commit 7: Split metadata
git add config/split_metadata.json
git commit -m "config: add train/val/test split metadata with normalization stats"

# Commit 8: CAE training summary
git add config/cae_training_summary.json
git commit -m "config: add CAE training summary (SSIM: 0.9756)"

# Commit 9: Classifier training summary
git add config/classifier_training_summary.json
git commit -m "config: add classifier training summary (F1: 0.9774)"

# Commit 10: Final evaluation results
git add config/final_evaluation_results.json config/threshold_recommendations.json
git commit -m "config: add final evaluation results and threshold recommendations"

# ============================================
# PHASE 3: TRAINING LOGS (Commits 11-14)
# ============================================

# Commit 11: CAE training history
git add logs/cae_training_history.csv
git commit -m "logs: add CAE training history (50 epochs)"

# Commit 12: Classifier training history
git add logs/classifier_training_history.csv
git commit -m "logs: add classifier training history (40 epochs, two-phase)"

# Commit 13: Validation classification report
git add logs/classification_report_val.csv
git commit -m "logs: add validation set classification report"

# Commit 14: Test classification report
git add logs/classification_report_test.csv
git commit -m "logs: add test set classification report (98.02% accuracy)"

# ============================================
# PHASE 4: NOTEBOOKS - DATA (Commits 15-18)
# ============================================

# Commit 15: Data exploration notebook
git add notebooks/1_Data_Exploration.ipynb
git commit -m "feat(notebook): add data exploration and EDA notebook"

# Commit 16: Data preprocessing notebook
git add notebooks/2_Data_Preprocessing.ipynb
git commit -m "feat(notebook): add data preprocessing and stratified splitting"

# Commit 17: Update notebook 1 (if any changes)
# git add notebooks/1_Data_Exploration.ipynb
# git commit -m "refactor(notebook): improve visualizations in data exploration"

# Commit 18: Add dataset download instructions
cat > data/README.md << 'DATAEOF'
# Dataset

## PlantVillage Dataset

Download the PlantVillage dataset and place it in `data/raw/color/`.

### Download Options:
1. [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. [GitHub](https://github.com/spMohanty/PlantVillage-Dataset)

### Expected Structure:
```
data/
├── raw/
│   └── color/
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       └── ...
└── processed/
    ├── train/
    ├── val/
    └── test/
```

Run Notebook 2 to create the processed splits.
DATAEOF
git add data/README.md
git commit -m "docs: add dataset download instructions"

# ============================================
# PHASE 5: NOTEBOOKS - CAE (Commits 19-22)
# ============================================

# Commit 19: CAE training notebook
git add notebooks/3_CAE_Training.ipynb
git commit -m "feat(notebook): add CAE self-supervised training notebook"

# Commit 20: CAE architecture documentation
cat > models/README.md << 'MODELEOF'
# Models

## Convolutional Autoencoder (CAE)
- `cae_encoder.pth` - Pre-trained encoder weights
- `cae_full.pth` - Full autoencoder model

## Classifier
- `classifier_final.pth` - Final trained classifier

## Production
- `production/model_config.json` - Model configuration
- `production/inference.py` - Standalone inference script
- `production/model_weights.pth` - Lightweight weights (not tracked)

### Training Results:
- CAE SSIM: 0.9756
- CAE PSNR: 40.62 dB
- Classifier F1: 0.9774
- Test Accuracy: 98.02%
MODELEOF
git add models/README.md
git commit -m "docs: add models directory documentation"

# Commit 21: Production inference script
git add models/production/inference.py
git commit -m "feat: add standalone CLI inference script"

# Commit 22: Production model config
git add models/production/model_config.json
git commit -m "config: add production model configuration"

# ============================================
# PHASE 6: NOTEBOOKS - CNN (Commits 23-26)
# ============================================

# Commit 23: CNN classifier notebook
git add notebooks/4_CNN_Classifier_Training.ipynb
git commit -m "feat(notebook): add CNN classifier with two-phase training"

# Commit 24: Inference notebook
git add notebooks/Inference_Notebook.ipynb
git commit -m "feat(notebook): add production inference pipeline"

# Commit 25: Evaluation notebook
git add notebooks/5_Threshold_Optimization_and_Evaluation.ipynb
git commit -m "feat(notebook): add threshold optimization and final evaluation"

# Commit 26: Workflow diagrams notebook
git add notebooks/Workflow_Diagrams.ipynb
git commit -m "feat(notebook): add workflow diagrams generation notebook"

# ============================================
# PHASE 7: OUTPUT FIGURES - TRAINING (Commits 27-32)
# ============================================

# Commit 27: EDA figures
git add outputs/fig_01*.png outputs/fig_02*.png outputs/fig_03*.png
git commit -m "docs(figures): add EDA and class distribution figures"

# Commit 28: Data split figures
git add outputs/fig_04*.png outputs/fig_05*.png outputs/fig_06*.png
git commit -m "docs(figures): add data split and sample visualization figures"

# Commit 29: More preprocessing figures
git add outputs/fig_07*.png outputs/fig_08*.png outputs/fig_09*.png outputs/fig_10*.png
git commit -m "docs(figures): add preprocessing and normalization figures"

# Commit 30: CAE training figures
git add outputs/fig_11*.png outputs/fig_12*.png outputs/fig_13*.png
git commit -m "docs(figures): add CAE training curves and reconstruction figures"

# Commit 31: Classifier training figures
git add outputs/fig_14*.png outputs/fig_15*.png
git commit -m "docs(figures): add classifier training curves and validation confusion matrix"

# Commit 32: Test evaluation figures
git add outputs/fig_16*.png outputs/fig_17*.png
git commit -m "docs(figures): add test set confusion matrix and per-class metrics"

# ============================================
# PHASE 8: OUTPUT FIGURES - EVALUATION (Commits 33-36)
# ============================================

# Commit 33: Threshold figures
git add outputs/fig_18*.png outputs/fig_19*.png
git commit -m "docs(figures): add threshold optimization and confidence distribution"

# Commit 34: ROC and PR curves
git add outputs/fig_20*.png outputs/fig_21*.png
git commit -m "docs(figures): add ROC and Precision-Recall curves"

# Commit 35: t-SNE and error analysis
git add outputs/fig_22*.png outputs/fig_23*.png
git commit -m "docs(figures): add t-SNE visualization and misclassification analysis"

# Commit 36: Inference sample figures
git add outputs/fig_inference*.png outputs/batch_predictions.csv 2>/dev/null
git commit -m "docs(figures): add inference sample predictions" 2>/dev/null || echo "No inference figures"

# ============================================
# PHASE 9: WORKFLOW DIAGRAMS (Commits 37-40)
# ============================================

# Commit 37: Project workflow diagram
git add outputs/diagram_01*.png outputs/diagram_02*.png
git commit -m "docs(diagrams): add project workflow and data pipeline diagrams"

# Commit 38: Architecture diagrams
git add outputs/diagram_03*.png outputs/diagram_04*.png
git commit -m "docs(diagrams): add CAE and CNN architecture diagrams"

# Commit 39: Training and inference diagrams
git add outputs/diagram_05*.png outputs/diagram_06*.png
git commit -m "docs(diagrams): add training and inference pipeline diagrams"

# Commit 40: Dashboard and class diagrams
git add outputs/diagram_07*.png outputs/diagram_08*.png
git commit -m "docs(diagrams): add performance dashboard and disease classes overview"

# ============================================
# PHASE 10: DOCUMENTATION (Commits 41-45)
# ============================================

# Commit 41: README
git add README.md
git commit -m "docs: add comprehensive README with results and usage"

# Commit 42: Notebooks README
cat > notebooks/README.md << 'NBEOF'
# Notebooks

Execute notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_Data_Exploration.ipynb` | Dataset analysis and EDA |
| 2 | `2_Data_Preprocessing.ipynb` | Stratified splitting |
| 3 | `3_CAE_Training.ipynb` | Self-supervised pre-training |
| 4 | `4_CNN_Classifier_Training.ipynb` | Two-phase classification |
| 5 | `5_Threshold_Optimization_and_Evaluation.ipynb` | Final evaluation |
| - | `Inference_Notebook.ipynb` | Production inference |
| - | `Workflow_Diagrams.ipynb` | Generate diagrams |

## Requirements
- GPU with 8GB+ VRAM recommended
- ~30 minutes total runtime
NBEOF
git add notebooks/README.md
git commit -m "docs: add notebooks README with execution order"

# Commit 43: Config README
cat > config/README.md << 'CFGEOF'
# Configuration Files

| File | Description |
|------|-------------|
| `dataset_config.json` | Dataset paths and class info |
| `split_metadata.json` | Train/val/test split info + normalization |
| `cae_training_summary.json` | CAE training results |
| `classifier_training_summary.json` | Classifier training results |
| `final_evaluation_results.json` | Test set evaluation |
| `threshold_recommendations.json` | Deployment thresholds |
CFGEOF
git add config/README.md
git commit -m "docs: add configuration files documentation"

# Commit 44: Outputs README
cat > outputs/README.md << 'OUTEOF'
# Output Files

## Training Figures (fig_01 - fig_15)
- EDA and class distribution
- Data preprocessing visualizations
- CAE training curves and reconstructions
- Classifier training curves

## Evaluation Figures (fig_16 - fig_23)
- Test set confusion matrix
- Per-class metrics
- Threshold optimization
- ROC and PR curves
- t-SNE visualization
- Error analysis

## Workflow Diagrams (diagram_01 - diagram_08)
- Project workflow
- Data pipeline
- CAE architecture
- CNN two-phase training
- Training pipeline
- Inference pipeline
- Performance dashboard
- Disease classes
OUTEOF
git add outputs/README.md
git commit -m "docs: add outputs directory documentation"

# Commit 45: Final cleanup
git add -A
git commit -m "chore: final cleanup and organization" 2>/dev/null || echo "Nothing to commit"

echo ""
echo "============================================"
echo "Total commits made:"
git rev-list --count HEAD
echo "============================================"

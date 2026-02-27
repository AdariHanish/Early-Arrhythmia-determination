1. Setup
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
# Install requirements
pip install -r requirements.txt

2. Run Quick Test
python main.py --mode test

3. Run Full Training
# Basic training
python main.py --mode train --epochs 100 --batch_size 32
# With baseline comparison
python main.py --mode train --epochs 100 --run_baselines
# With cross-validation
python main.py --mode train --epochs 100 --run_cv --run_baselines

4. Run Inference Only
python main.py --mode inference

5. Run Baseline Comparison Only
python main.py --mode baseline --epochs 50

# ============================================================================
# Run this cell in Google Colab to execute the complete pipeline
# ============================================================================

# Step 1: Upload files to Colab
# - Upload all .py files to /content/
# - Upload dataset CSVs to /content/dataset/

# Step 2: Install requirements
!pip install -r requirements.txt

# Step 3: Run the pipeline
# Option A: MIT-BIH dataset (5-class arrhythmia)
!python main.py --dataset mitbih

# Option B: PTB dataset (binary classification)
# !python main.py --dataset ptbdb

# Option C: Both datasets
# !python main.py --dataset both

# Step 4: View results
from IPython.display import Image, display
import os

results_dir = "/content/results"
for f in sorted(os.listdir(results_dir)):
    if f.endswith('.png'):
        print(f"\n{'='*50}")
        print(f"  {f}")
        print(f"{'='*50}")
        display(Image(os.path.join(results_dir, f)))
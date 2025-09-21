"""
🛍️ RETAIL SALES FORECASTING - CONFIGURATION (Conda Version)
Project configuration and directory setup for Conda environment
"""

import os
import sys

print("🚀 INITIALIZING RETAIL SALES FORECASTING PROJECT")
print("=" * 55)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
NOTEBOOKS_PATH = os.path.join(PROJECT_ROOT, 'notebooks')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Forecasting parameters
FORECAST_HORIZON = 30  # days
LAG_FEATURES = [1, 7, 14, 30]  # lag days
ROLLING_WINDOWS = [7, 14, 30]  # rolling average windows

# Create all directories
directories = [
    DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, 
    MODELS_PATH, RESULTS_PATH, NOTEBOOKS_PATH, SRC_PATH,
    os.path.join(RESULTS_PATH, 'plots'),
    os.path.join(RESULTS_PATH, 'forecasts')
]

print("📁 Creating directory structure...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"   ✅ {os.path.relpath(directory)}")

# Create src/__init__.py to make it a package
init_file = os.path.join(SRC_PATH, '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write('# Retail Sales Forecasting Package\n')
    print(f"   ✅ Created {os.path.relpath(init_file)}")

# Environment verification
print(f"\n🔧 ENVIRONMENT VERIFICATION:")
print(f"   Python version: {sys.version.split()[0]}")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'Not detected')}")

# Check critical packages
critical_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'flask']
missing_packages = []

for package in critical_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  MISSING PACKAGES:")
    print(f"   Run: conda install {' '.join(missing_packages)}")
else:
    print(f"\n🎉 PROJECT SETUP COMPLETED SUCCESSFULLY!")
    print(f"📊 Ready to start the forecasting pipeline!")

print(f"\n📋 NEXT STEPS:")
print(f"   1. Run: python 01_generate_data.py")
print(f"   2. Run: python 02_preprocess_data.py")
print(f"   3. Run: python 03_run_eda.py")
print(f"   4. Run: python 04_train_models.py")
print(f"   5. Run: python 05_deployment_api.py")
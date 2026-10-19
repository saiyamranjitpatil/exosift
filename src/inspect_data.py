from pathlib import Path
import pandas as pd


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_PATH = BASE_DIR / "data" / "raw" / "exoTrain.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "exoTest.csv"


print("\n🔭 EXOSIFT — DATASET INSPECTION")
print("=" * 50)


# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


# Basic shapes
print("\n📦 DATASET SIZE")
print("-" * 50)
print(f"Training data : {train.shape}")
print(f"Testing data  : {test.shape}")


# Columns
print("\n🧱 COLUMN STRUCTURE")
print("-" * 50)
print(f"Total columns: {len(train.columns)}")
print(f"First 5 columns: {train.columns[:5].tolist()}")
print(f"Last 5 columns: {train.columns[-5:].tolist()}")


# Labels
print("\n🎯 LABEL DISTRIBUTION")
print("-" * 50)
print(train["LABEL"].value_counts().sort_index())


# Missing values
print("\n🕳️ MISSING VALUES")
print("-" * 50)
print(f"Training: {train.isna().sum().sum()}")
print(f"Testing : {test.isna().sum().sum()}")


# Sample labels
print("\n🏷️ FIRST 10 LABELS")
print("-" * 50)
print(train["LABEL"].head(10).tolist())


# Sample flux values
print("\n🌟 FIRST STAR — FIRST 10 FLUX VALUES")
print("-" * 50)
print(train.iloc[0, 1:11].tolist())


print("\n" + "=" * 50)
print("✅ Dataset inspection complete.")
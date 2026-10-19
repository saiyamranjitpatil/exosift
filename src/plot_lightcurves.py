from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "raw" / "exoTrain.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(TRAIN_PATH)

flux_columns = [column for column in df.columns if column.startswith("FLUX.")]


# --------------------------------------------------
# Separate classes
# --------------------------------------------------

planet = df[df["LABEL"] == 2]
non_planet = df[df["LABEL"] == 1]


print(f"Planet examples     : {len(planet)}")
print(f"Non-planet examples : {len(non_planet)}")


# --------------------------------------------------
# Plot 1 — Planet candidate
# --------------------------------------------------

planet_curve = planet.iloc[0][flux_columns].values

plt.figure(figsize=(14, 5))
plt.plot(planet_curve)
plt.title("ExoSift — Example Exoplanet Light Curve")
plt.xlabel("Time Step")
plt.ylabel("Flux")
plt.grid(True)
plt.show()


# --------------------------------------------------
# Plot 2 — Non-planet
# --------------------------------------------------

normal_curve = non_planet.iloc[0][flux_columns].values

plt.figure(figsize=(14, 5))
plt.plot(normal_curve)
plt.title("ExoSift — Example Non-Planet Light Curve")
plt.xlabel("Time Step")
plt.ylabel("Flux")
plt.grid(True)
plt.show()


# --------------------------------------------------
# Plot 3 — Compare several examples
# --------------------------------------------------

plt.figure(figsize=(14, 6))

for i in range(5):
    plt.plot(non_planet.iloc[i][flux_columns].values)

plt.title("ExoSift — Several Non-Planet Light Curves")
plt.xlabel("Time Step")
plt.ylabel("Flux")
plt.grid(True)
plt.show()
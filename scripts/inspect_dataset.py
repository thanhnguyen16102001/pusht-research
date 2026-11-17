import numpy as np
import zarr

# 1. Load dataset
dataset_path = "data/demonstrations/demo_actions_v1.zarr"

root = zarr.open_group(
    dataset_path,
    mode="r"
)

data = root["data"]
ends = np.asarray(root["meta"]["episode_ends"][:])

# 2. Count episodes and steps
num_episodes = len(ends)
num_steps = data["action"].shape[0]

print("\n========== DATASET SUMMARY ==========")
print("Dataset  :", dataset_path)
print("Episodes :", num_episodes)
print("Steps    :", num_steps)

# 3. Inspect arrays
print("\n========== DATA STRUCTURE ==========")

for name in data.array_keys():
    arr = data[name]

    print(
        f"{name:12s} "
        f"shape={str(arr.shape):25s} "
        f"dtype={arr.dtype}"
    )

# 4. Verify episode boundaries
assert num_episodes > 0, "No episodes found"
assert np.all(np.diff(ends) > 0), "Invalid episode boundaries"
assert int(ends[-1]) == num_steps, "Step count mismatch"

# 5. Episode statistics
starts = np.concatenate(([0], ends[:-1]))
lengths = ends - starts

print("\n========== EPISODE STATISTICS ==========")
print("Min steps :", lengths.min())
print("Max steps :", lengths.max())
print("Avg steps :", round(lengths.mean(), 2))

for i, length in enumerate(lengths):
    print(f"Episode {i:3d}: {length} steps")

# 6. Check first sample
print("\n========== FIRST SAMPLE ==========")
print("State   :", data["state"][0])
print("Action  :", data["action"][0])
print("Image   :", data["img"][0].shape)

print("\n[OK] Dataset basic validation passed.")

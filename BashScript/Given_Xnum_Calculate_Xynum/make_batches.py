from pathlib import Path
import sys
import math

# ================= Parameter settings =================
NUM_BATCHES = 4       # Maximum number of batches per richness
BATCH_SIZE = 64       # Maximum number of files per batch
# ======================================================

data_dir = Path(sys.argv[1])
files = sorted(data_dir.glob("*.csv"))
groups = {}

for f in files:
    richness = int(f.stem.split("-")[0])
    groups.setdefault(richness, []).append(f.name)

Path("batch_lists").mkdir(exist_ok=True)

master_list = []

for richness in sorted(groups):

    flist = groups[richness]
    n_files = len(flist)

    required_batches = math.ceil(n_files / BATCH_SIZE)

    # Check whether the specified number of batches is sufficient
    if required_batches > NUM_BATCHES:
        raise ValueError(
            f"richness={richness} has {n_files} files, "
            f"but {NUM_BATCHES} batches can only hold "
            f"{NUM_BATCHES * BATCH_SIZE} files."
        )

    print(f"richness={richness}, files={n_files}")

    for batch_id in range(NUM_BATCHES):

        start = batch_id * BATCH_SIZE
        end = min((batch_id + 1) * BATCH_SIZE, n_files)

        # Skip empty batches
        if start >= n_files:
            break

        outfile = Path("batch_lists") / f"{richness}_batch{batch_id}.txt"

        with open(outfile, "w") as fw:
            for x in flist[start:end]:
                fw.write(x + "\n")

        master_list.append(str(outfile))

        print(f"  batch {batch_id}: {end - start} files")

# Generate the master batch list
with open("master_batch_list.txt", "w") as fw:
    for x in master_list:
        fw.write(x + "\n")
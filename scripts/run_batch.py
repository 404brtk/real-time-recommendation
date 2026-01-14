import subprocess
import sys
import time

STEPS = [
    ("Convert CSV to Delta", "src.batch.csv_to_delta"),
    ("Preprocess data", "src.batch.preprocess"),
    ("Train ALS model", "src.batch.train_als"),
    ("Load to Redis/Qdrant", "src.batch.loader"),
    ("Calculate popular items", "src.batch.calc_popular"),
]


def run_step(name: str, module: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Step: {name}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(["uv", "run", "-m", module])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFailed: {name} (exit code {result.returncode})")
        return False

    print(f"\nCompleted: {name} in {elapsed:.1f}s")
    return True


def main():
    print("Starting batch pipeline...")
    total_start = time.time()

    for name, module in STEPS:
        if not run_step(name, module):
            print("\nPipeline failed!")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Batch pipeline complete! Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

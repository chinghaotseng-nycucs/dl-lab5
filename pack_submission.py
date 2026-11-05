"""Build LAB5_<StudentID>.zip exactly as spec section 5 lists it. Extra, not required as a file.

Usage:
    python pack_submission.py --task1 runs/task1/best_model.pt --task2 runs/task2/best_model.pt \
        --task3-run runs/task3 --report ../LAB5_315551118.pdf --video ../LAB5_315551118.mp4

It stops if anything is missing, and it refuses to overwrite an existing ZIP (a wrong name or folder costs 5 points).
"""

import argparse
import hashlib
import os
import sys
import zipfile

STUDENT_ID = "315551118"
# Spec 5: only source code goes in the Code folder, so README.md and the test helpers stay out.
CODE_FILES = [
    "dqn.py",
    "run_experiment.py",
    "eval_snapshot.py",
    "test_model.py",
    "test_model_task1.py",
    "test_model_task2.py",
    "test_model_task3.py",
    "make_plots.py",
    "experiments.sh",
    "requirements.txt",
]
TASK3_STEPS = [600000, 1000000, 1500000, 2000000, 2500000]  # spec 4.2.2 and 5


def sha256_of(path):
    """Return the SHA-256 of a file, as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # read 1 MB at a time until f.read returns b""
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build the Lab5 submission ZIP")
    parser.add_argument("--task1", required=True)  # no default: the best Task 1 snapshot
    parser.add_argument("--task2", required=True)  # no default: the best Task 2 snapshot
    parser.add_argument("--task3-run", required=True)  # no default: the run folder with snapshot_<n>.pt
    parser.add_argument("--task3-best", default="")  # default: best_model.pt in the Task 3 run folder
    parser.add_argument("--report", required=True)  # no default: the PDF
    parser.add_argument("--video", required=True)  # no default: the MP4, 5 to 6 minutes
    parser.add_argument("--out-dir", default="..")  # default .. : next to src/
    args = parser.parse_args()

    prefix = f"LAB5_{STUDENT_ID}"
    code_folder = f"{prefix}_Code"
    best = args.task3_best or os.path.join(args.task3_run, "best_model.pt")
    # (file on disk, path inside the ZIP), in the order of the spec's tree
    items = [(name, f"{code_folder}/{name}") for name in CODE_FILES]
    items += [(args.report, f"{prefix}.pdf"), (args.video, f"{prefix}.mp4")]
    items += [(args.task1, f"{prefix}_task1.pt"), (args.task2, f"{prefix}_task2.pt")]
    items += [(os.path.join(args.task3_run, f"snapshot_{n}.pt"), f"{prefix}_task3_{n}.pt") for n in TASK3_STEPS]
    items += [(best, f"{prefix}_task3_best.pt")]

    missing = [source for source, _ in items if not os.path.isfile(source)]
    if missing:
        sys.exit("Missing files, nothing written:\n  " + "\n  ".join(missing))
    out = os.path.join(args.out_dir, f"{prefix}.zip")
    if os.path.exists(out):
        sys.exit(f"{out} already exists. Move it away first, so that no earlier result is overwritten.")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, inside in items:
            archive.write(source, inside)

    for source, inside in items:
        print(f"{sha256_of(source)[:16]}  {os.path.getsize(source):>11,} B  {inside}")
    print(f"\nWrote {out}: {os.path.getsize(out) / 1024**2:.1f} MB, SHA-256 {sha256_of(out)}")
    print("The script does not check the video length. It must be 5 to 6 minutes (spec 4.2.1).")


if __name__ == "__main__":
    main()

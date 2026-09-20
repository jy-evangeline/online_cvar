"""Build a numeric-only ("slim") copy of a toxicity-score data directory.

online_cvar_detoxify_ru.py reads only four things out of these pickles:

  * ``detoxify_ft``                  -- machine (fine-tuned Detoxify) scores
  * ``detoxify_human["toxicity"]``   -- human toxicity scores
  * the *indices* in ``conformal[key]["set"]`` -- the response texts there are
    discarded by distortion_risk_control_online()
  * ``pred`` is stored by load_x_cal() but never read afterwards

Everything else (the generated responses themselves, perplexities, the other
Detoxify heads) is dead weight -- and it is ~99% of the bytes, which is what
puts the raw directory over GitHub's limits.

This script rewrites each pickle keeping only the fields above, in the *same*
nested structure, so online_cvar_detoxify_ru.py runs against the output with no
code change and produces identical numbers. Float values are copied by
reference, never re-encoded, so they are bit-for-bit identical.

Usage:
    python build_slim_data.py --src data/llama3.2_real_toxic \
                              --dst data_slim/llama3.2_real_toxic
"""

import argparse
import os
import pickle


def slim_responses(src_path: str, dst_path: str) -> None:
    """Keep only detoxify_ft and detoxify_human['toxicity'] per prompt."""
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    out = {}
    for key, entry in data.items():
        ft = entry.get("detoxify_ft")
        human = entry.get("detoxify_human")
        if ft is None or human is None:
            continue
        tox = human.get("toxicity") if hasattr(human, "get") else None
        if tox is None:
            continue
        # Objects are referenced, not rebuilt -> identical float values.
        out[key] = {"detoxify_ft": ft, "detoxify_human": {"toxicity": tox}}

    with open(dst_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def slim_conformal(src_path: str, dst_path: str) -> None:
    """Keep the conformal-set response indices, drop the response texts."""
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    out = {}
    for key, entry in data.items():
        pairs = entry["set"]
        # The consumer does `[idx for idx, _ in C_all]`, so the second slot only
        # has to exist. "" keeps the tuple shape without carrying the text.
        out[key] = {"set": [(idx, "") for idx, _ in pairs]}

    with open(dst_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Directory of raw .pkl files")
    p.add_argument("--dst", required=True, help="Directory to write slim .pkl files to")
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    names = sorted(n for n in os.listdir(args.src) if n.endswith(".pkl"))

    for name in names:
        src_path = os.path.join(args.src, name)
        dst_path = os.path.join(args.dst, name)
        if name.startswith("conformal_set"):
            slim_conformal(src_path, dst_path)
        else:
            slim_responses(src_path, dst_path)
        before = os.path.getsize(src_path) / 1e6
        after = os.path.getsize(dst_path) / 1e6
        print(f"{name}: {before:8.1f} MB -> {after:6.2f} MB")

    total_before = sum(os.path.getsize(os.path.join(args.src, n)) for n in names) / 1e6
    total_after = sum(os.path.getsize(os.path.join(args.dst, n)) for n in names) / 1e6
    print(f"\ntotal: {total_before:.1f} MB -> {total_after:.2f} MB "
          f"({total_before / max(total_after, 1e-9):.0f}x smaller)")


if __name__ == "__main__":
    main()

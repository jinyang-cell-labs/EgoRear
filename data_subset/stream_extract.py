# Stream-extract a single test sequence from the Ego4View-RW split tar.gz
# archives WITHOUT storing the archives (disk is tight). Reads the
# concatenated gzip stream on stdin, extracts:
#   - every *_metadata.json (tiny, needed by the dataloader)
#   - all files of the FIRST test sequence directory encountered (seq_2-*)
# and exits as soon as both are complete, breaking the upstream curl pipe.

import re
import sys
import tarfile

OUT = "/workspace_out"  # overridden below when run directly
TEST_SEQ_RE = re.compile(r"/(seq_2-\d+)/")


def main(out_dir):
    tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")
    target = None
    extracted = 0
    have_metadata = False
    seen_after_target = 0
    for m in tf:
        if m.name.endswith("_metadata.json"):
            tf.extract(m, out_dir)
            have_metadata = True
            print(f"[meta] {m.name}", flush=True)
            if target and seen_after_target:
                break
            continue
        match = TEST_SEQ_RE.search(m.name)
        if match is None:
            continue
        seq = match.group(1)
        if target is None:
            target = seq
            print(f"[target] {seq}", flush=True)
        if seq == target:
            tf.extract(m, out_dir)
            extracted += 1
            if extracted % 200 == 0:
                print(f"[extract] {extracted} files ...", flush=True)
        else:
            seen_after_target += 1
            if have_metadata:
                break
            if seen_after_target > 8000:  # metadata not adjacent; give up waiting
                break
    print(f"[done] target={target} files={extracted} metadata={have_metadata}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

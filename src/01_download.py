"""Download ridership chunks from the data.dubai dataset-download API (datasetId 466403).

API facts: 30 chunk folders per page, each with a .csv.gz + .json.gz and a SIGNED url that
expires after ~600 s -> the page is re-requested for fresh links (cached <= 4 min). Each chunk
transfers as ~8 MB gzip (the listed 154 MB is the UNCOMPRESSED size). The CDN may send
`Content-Encoding: gzip`, so raw bytes are saved and the file is named by its real content
(.csv.gz for gzip, .csv for plain text; an HTML/JSON error page stops with a message).

  python src/01_download.py --build-index          # also prints which exports (timestamps) exist
  python src/01_download.py --index-summary
  python src/01_download.py --get 1 1552 3104
  python src/01_download.py --get-all --stream      # all indexed chunk IDs, including non-contiguous IDs; resumable
"""
import argparse, json, os, shutil, sys, time
from collections import defaultdict
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

API = "https://data.dubai/o/dda/data-services/dataset-download"
DATASET_ID = 466403
PAGE_SIZE = 30
INDEX_FILE = C.INTERIM_DIR / "chunk_index.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def headers():
    h = {"Accept": "application/json, */*", "User-Agent": UA}
    if os.getenv("DATA_DUBAI_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['DATA_DUBAI_TOKEN']}"
    if os.getenv("DATA_DUBAI_COOKIE"):
        h["Cookie"] = os.environ["DATA_DUBAI_COOKIE"]
    return h


def fetch_page(page, retries=5):
    """One listing page, with retries. On a non-JSON reply (empty body, firewall page, rate limit)
    it prints the HTTP status, content type and the first bytes, waits (2,4,8,... s) and retries."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(API, params=dict(datasetId=DATASET_ID, page=page, pageSize=PAGE_SIZE,
                                              sortDir="desc"), headers=headers(), timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]!r}")
            r.raise_for_status()
            try:
                j = r.json()
            except ValueError:
                raise RuntimeError(f"non-JSON reply (HTTP {r.status_code}, "
                                   f"type {r.headers.get('Content-Type')}): {r.text[:200]!r}")
            if not j.get("success"):
                raise RuntimeError(f"API said: {j.get('message')}")
            return j["data"]["metadata"], j["data"]["pagination"]
        except (requests.RequestException, RuntimeError) as e:
            last = e
            wait = 2 ** (attempt + 1)
            print(f"  page {page}: {e} -> retry in {wait}s ({attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError(f"page {page} failed after {retries} attempts: {last}. If it is a 'Request Rejected' "
                       "page, wait a few minutes, or copy your browser's Cookie header into DATA_DUBAI_COOKIE.")


def chunk_no(folder: str) -> int:
    return int(folder.rsplit("_", 1)[1])


def csv_entry(meta_item):
    return next(f for f in meta_item["files"] if f["file_extension"] == "csv")


def stem_of(file_name: str) -> str:
    return file_name.split(".")[0]


def summarize(index):
    by_export = defaultdict(list)
    for n, e in index.items():
        by_export[e["folder"].rsplit("_", 1)[0]].append(n)
    print(f"{len(index)} chunks indexed; exports (folder prefix = export timestamp):")
    for prefix, ns in sorted(by_export.items()):
        print(f"  {prefix}: {len(ns)} chunks, numbers {min(ns)}..{max(ns)}")
    if len(by_export) > 1:
        print("  NOTE: several exports exist. Check whether they overlap before combining them.")


def build_index():
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    index, page, collisions, total = {}, 1, [], None
    while True:
        md, pg = fetch_page(page)
        total = pg["total"]
        for m in md:
            f, n = csv_entry(m), chunk_no(m["file_folder"])
            if n in index and index[n]["folder"] != m["file_folder"]:
                collisions.append((n, index[n]["folder"], m["file_folder"]))
            index[n] = {"page": page, "folder": m["file_folder"], "file": f["file_name"], "bytes": f["file_size"]}
        print(f"page {page}/{pg['page_count']}: {len(index)} chunks indexed")
        if not pg["has_more"]:
            break
        page += 1
        time.sleep(0.3)                                  # be gentle: avoid tripping the firewall
    INDEX_FILE.write_text(json.dumps(index, indent=1))
    summarize(index)
    if collisions:
        print(f"WARNING: {len(collisions)} chunk numbers appear in more than one export (later entry kept): {collisions[:3]}")
    if len(index) != total:
        print(f"WARNING: API total is {total} but {len(index)} distinct chunk numbers were indexed.")
    return index


def load_index():
    if INDEX_FILE.exists():
        return {int(k): v for k, v in json.loads(INDEX_FILE.read_text()).items()}
    return build_index()


def download(url: str, stem: str, expected_bytes: int = 0, raw_dir: Path = None) -> Path:
    """Fetch one chunk; return the final path (.csv.gz or .csv, decided by content)."""
    raw_dir = raw_dir or C.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    gz, plain = raw_dir / f"{stem}.csv.gz", raw_dir / f"{stem}.csv"
    for existing in (gz, plain):      # written as .part then renamed, so existence == complete
        if existing.exists() and existing.stat().st_size > 0:
            print(f"  already have {existing.name}")
            return existing
    tmp = raw_dir / f"{stem}.part"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        r.raw.decode_content = False                   # keep the bytes exactly as sent
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, 1 << 20)
    with open(tmp, "rb") as f:
        head = f.read(512)
    if head[:2] == b"\x1f\x8b":
        final = gz
    elif head.lstrip()[:1] in (b"<", b"{"):
        tmp.unlink()
        raise RuntimeError(f"Got an error page instead of data (link expired?): {head[:200]!r}")
    else:
        final = plain
    tmp.replace(final)
    return final


def get_chunks(numbers, stream=False, keep=False):
    index = load_index()
    numbers = list(numbers)
    con = None
    if stream:
        import aggcore
        con = aggcore.connect()
    cache, t0, fresh = {}, time.time(), 0
    for i, n in enumerate(numbers, 1):
        info = index[n]
        stem = stem_of(info["file"])
        if stream and aggcore.is_done(stem, con):
            continue
        page = info["page"]
        if page not in cache or time.time() - cache[page][0] > 240:     # signed links last ~10 min
            cache = {page: (time.time(), fetch_page(page)[0])}
        item = next(m for m in cache[page][1] if chunk_no(m["file_folder"]) == n)
        f = csv_entry(item)
        path = download(f["file_url"], stem)
        if stream:
            aggcore.add_chunk(path, con)
            if not keep:
                path.unlink()
        fresh += 1
        el = time.time() - t0
        print(f"[{i}/{len(numbers)}] chunk {n} ok | {el / fresh:.1f}s/chunk | "
              f"ETA {(len(numbers) - i) * el / fresh / 60:.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--index-summary", action="store_true")
    ap.add_argument("--get", type=int, nargs="+")
    ap.add_argument("--get-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    ap.add_argument("--get-all", action="store_true", help="process every chunk ID in the index")
    ap.add_argument("--stream", action="store_true", help="add each chunk to the accumulator, then delete the raw file")
    ap.add_argument("--keep", action="store_true", help="with --stream: keep the raw file")
    a = ap.parse_args()
    if a.build_index:
        build_index()
    if a.index_summary:
        summarize(load_index())
    if a.get:
        get_chunks(a.get, a.stream, a.keep)
    if a.get_range:
        get_chunks(range(a.get_range[0], a.get_range[1] + 1), a.stream, a.keep)
    if a.get_all:
        get_chunks(sorted(load_index()), a.stream, a.keep)

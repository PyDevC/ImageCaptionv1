import urllib.request
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_DIR = Path("COCO")
IMAGES_DIR = BASE_DIR / "images"

MAX_WORKERS = 4

FILES = {
    "train2017.zip": (
        "http://images.cocodataset.org/zips/train2017.zip",
        IMAGES_DIR,
    ),
    "val2017.zip": (
        "http://images.cocodataset.org/zips/val2017.zip",
        IMAGES_DIR,
    ),
    "test2017.zip": (
        "http://images.cocodataset.org/zips/test2017.zip",
        IMAGES_DIR,
    ),
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        BASE_DIR,
    ),
}


def download_file(name, url, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / name

    if output_path.exists():
        print(f"[SKIP] {name} already exists")
        return output_path

    print(f"[START] {name}")

    try:
        urllib.request.urlretrieve(
            url,
            output_path,
            reporthook=lambda block, size, total:
                progress(name, block, size, total)
        )

        print(f"\n[DONE] {name}")

        return output_path

    except Exception:
        if output_path.exists():
            output_path.unlink()

        raise


def progress(name, block, block_size, total_size):

    if total_size <= 0:
        return

    downloaded = block * block_size
    percent = min(downloaded * 100 / total_size, 100)

    print(
        f"\r{name:35} {percent:6.2f}%",
        end="",
        flush=True
    )


def extract_file(zip_path, destination):

    print(f"\n[EXTRACT] {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)

    print(f"[DONE] Extracted {zip_path.name}")


def main():

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(FILES)} files")
    print(f"Using {MAX_WORKERS} threads\n")

    downloaded_files = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(
                download_file,
                name,
                url,
                output_dir
            ): name

            for name, (url, output_dir) in FILES.items()
        }

        for future in as_completed(futures):

            name = futures[future]

            try:
                path = future.result()
                downloaded_files[name] = path

            except Exception as e:
                print(f"\n[ERROR] {name}: {e}")
                raise

    print("\n\nAll downloads complete.")
    print("Extracting...\n")

    for name, (_, output_dir) in FILES.items():

        zip_path = output_dir / name

        extract_file(
            zip_path,
            output_dir
        )

    print(f"Data Location: {BASE_DIR.resolve()}")


if __name__ == "__main__":
    main()

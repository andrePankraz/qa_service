'''
This file was created by ]init[ AG 2022.

Module for Downloading with caching.
'''
from datetime import datetime
from email.utils import parsedate_to_datetime, formatdate
import os
import requests
import shutil
from tqdm.auto import tqdm


def download(url: str, destination_file: str, desc: str | None = None) -> bool:
    headers = {}

    if os.path.exists(destination_file):
        mtime = os.path.getmtime(destination_file)
        headers["if-modified-since"] = formatdate(mtime, usegmt=True)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=10) as r:
            r.raise_for_status()
            if r.status_code == requests.codes.not_modified:
                return False
            if r.status_code == requests.codes.ok:
                total_length = r.headers.get("Content-Length")
                r.raw.decode_content = True  # decode stream with content-encoding: gzip
                # implement progress bar via tqdm
                with tqdm.wrapattr(r.raw, "read", total=int(total_length), desc=desc) if total_length is not None else r.raw as raw:
                    with open(destination_file, "wb") as f:
                        shutil.copyfileobj(raw, f)
                    if last_modified := r.headers.get("last-modified"):
                        new_mtime = parsedate_to_datetime(last_modified).timestamp()
                        os.utime(destination_file, times=(datetime.now().timestamp(), new_mtime))
                    return True
    except requests.exceptions.RequestException as err:
        if not os.path.exists(destination_file):
            raise SystemExit(err)
    return False


if __name__ == '__main__':
    '''
    Just for debugging.
    '''
    imports_folder = os.environ.get('IMPORTS_FOLDER', '/opt/qa_service/imports/')
    download("https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles-multistream.xml.bz2",
             imports_folder + "dewiki-latest-pages-articles-multistream.xml.bz2", "Downloading Wikipedia-Export")

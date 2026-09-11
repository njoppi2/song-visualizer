"""Serve review artifacts locally, including single-byte-range responses.

Malformed and multi-range ``Range`` headers are deliberately ignored and use
the normal full-file ``200`` response.  Syntactically valid but unsatisfiable
single ranges receive ``416 Range Not Satisfiable``.
"""

from __future__ import annotations

import argparse
import os
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO


DEFAULT_DIRECTORY = Path("outputs/reviews")
DEFAULT_PORT = 8770
_COPY_CHUNK_SIZE = 64 * 1024
_UNSATISFIABLE = object()


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    """A directory-scoped static-file handler with single Range support."""

    protocol_version = "HTTP/1.1"
    # ``SimpleHTTPRequestHandler.index_pages`` was introduced after Python
    # 3.10.  Define it here so directory index behavior is consistent across
    # the project's supported runtimes.
    index_pages = ("index.html", "index.htm")

    def log_message(self, format: str, *args: object) -> None:
        """Keep the review server quiet unless a caller opts into logging."""

    @staticmethod
    def _parse_range(value: str | None, size: int) -> tuple[int, int] | object | None:
        """Return an inclusive range, an unsatisfiable sentinel, or ``None``.

        ``None`` means the header is malformed or unsupported and should fall
        back to ordinary SimpleHTTPRequestHandler-style full-file serving.
        """
        if value is None:
            return None
        unit, separator, specification = value.partition("=")
        if separator != "=" or unit.strip().lower() != "bytes":
            return None
        specification = specification.strip()
        if not specification or "," in specification or specification.count("-") != 1:
            return None

        first, last = specification.split("-", 1)
        if not first:
            if not last.isascii() or not last.isdigit():
                return None
            suffix_length = int(last)
            if suffix_length == 0 or size == 0:
                return _UNSATISFIABLE
            return max(0, size - suffix_length), size - 1

        if not first.isascii() or not first.isdigit():
            return None
        start = int(first)
        if start >= size:
            return _UNSATISFIABLE
        if not last:
            return start, size - 1
        if not last.isascii() or not last.isdigit():
            return None
        end = int(last)
        if end < start:
            return None
        return start, min(end, size - 1)

    def send_head(self) -> BinaryIO | None:
        """Send headers and return a positioned file for GET requests.

        Path translation is intentionally delegated to the standard-library
        implementation, including its directory redirects and listings.
        """
        self._range_remaining: int | None = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            parts = urllib.parse.urlsplit(self.path)
            if not parts.path.endswith(("/", "%2f", "%2F")):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                new_parts = (parts.scheme, parts.netloc, parts.path + "/", parts.query, parts.fragment)
                self.send_header("Location", urllib.parse.urlunsplit(new_parts))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            for index in self.index_pages:
                index_path = os.path.join(path, index)
                if os.path.isfile(index_path):
                    path = index_path
                    break
            else:
                return self.list_directory(path)

        content_type = self.guess_type(path)
        if path.endswith("/"):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            file = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            size = os.fstat(file.fileno()).st_size
            byte_range = self._parse_range(self.headers.get("Range"), size)
            if byte_range is _UNSATISFIABLE:
                file.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None

            self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
            self.send_header("Content-type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            if byte_range:
                start, end = byte_range
                length = end - start + 1
                file.seek(start)
                self._range_remaining = length
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                length = size
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(os.fstat(file.fileno()).st_mtime))
            self.end_headers()
            return file
        except BaseException:
            file.close()
            raise

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        """Copy only the selected bytes and quietly tolerate client disconnects."""
        remaining = self._range_remaining
        try:
            while remaining is None or remaining > 0:
                chunk = source.read(_COPY_CHUNK_SIZE if remaining is None else min(_COPY_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class ReviewServer(ThreadingHTTPServer):
    """Thread-per-request local server suitable for browser review sessions."""

    daemon_threads = True
    allow_reuse_address = True


def create_server(directory: Path, port: int = DEFAULT_PORT) -> ReviewServer:
    """Create, but do not start, a localhost-only review server."""
    root = Path(directory).resolve()
    handler = partial(ReviewRequestHandler, directory=str(root))
    return ReviewServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve SongViz review artifacts locally.")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"review directory does not exist: {directory}")

    with create_server(directory, args.port) as server:
        host, port = server.server_address
        print(f"Serving {directory} at http://{host}:{port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from http.client import HTTPConnection
from threading import Thread

import pytest

from songviz.review_server import create_server


@pytest.fixture
def review_server(tmp_path):
    content = b"0123456789abcdef"
    (tmp_path / "review.txt").write_bytes(content)
    subdirectory = tmp_path / "subdir"
    subdirectory.mkdir()
    (subdirectory / "index.html").write_text("<h1>Rendered review</h1>", encoding="utf-8")
    server = create_server(tmp_path, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield host, port, content
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(server, method="GET", path="/review.txt", headers=None):
    host, port, _ = server
    connection = HTTPConnection(host, port, timeout=2)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    result = response.status, dict(response.getheaders()), body
    connection.close()
    return result


def test_full_file_and_query_path_use_normal_mime_type(review_server):
    status, headers, body = _request(review_server, path="/review.txt?cache_bust=1")

    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Length"] == "16"
    assert headers["Content-type"].startswith("text/plain")
    assert body == review_server[2]


def test_directory_index_page_is_served(review_server):
    status, headers, body = _request(review_server, path="/subdir/")

    assert status == 200
    assert headers["Content-type"].startswith("text/html")
    assert body == b"<h1>Rendered review</h1>"


def test_closed_and_open_ranges_return_only_requested_bytes(review_server):
    status, headers, body = _request(review_server, headers={"Range": "bytes=2-6"})
    assert (status, headers["Content-Range"], headers["Content-Length"], body) == (
        206,
        "bytes 2-6/16",
        "5",
        b"23456",
    )

    status, headers, body = _request(review_server, headers={"Range": "bytes=10-"})
    assert (status, headers["Content-Range"], headers["Content-Length"], body) == (
        206,
        "bytes 10-15/16",
        "6",
        b"abcdef",
    )


def test_suffix_range_and_head_range(review_server):
    status, headers, body = _request(review_server, headers={"Range": "bytes=-4"})
    assert (status, headers["Content-Range"], headers["Content-Length"], body) == (
        206,
        "bytes 12-15/16",
        "4",
        b"cdef",
    )

    status, headers, body = _request(review_server, method="HEAD", headers={"Range": "bytes=2-6"})
    assert (status, headers["Content-Range"], headers["Content-Length"], body) == (
        206,
        "bytes 2-6/16",
        "5",
        b"",
    )


@pytest.mark.parametrize("range_header", ["bytes=nope", "bytes=0-1,3-4"])
def test_invalid_or_multi_ranges_fall_back_to_full_file(review_server, range_header):
    status, headers, body = _request(review_server, headers={"Range": range_header})

    assert status == 200
    assert "Content-Range" not in headers
    assert headers["Content-Length"] == "16"
    assert body == review_server[2]


def test_unsatisfiable_range_returns_416_with_total_size(review_server):
    status, headers, body = _request(review_server, headers={"Range": "bytes=16-"})

    assert status == 416
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Range"] == "bytes */16"
    assert headers["Content-Length"] == "0"
    assert body == b""

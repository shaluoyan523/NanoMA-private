"""Trace viewer: HTTP server for the single-page trace debugger UI."""

import http.server
import json
import math
import os
import sys
import time
import threading
from pathlib import Path
from urllib.parse import unquote

LOG_DIR = Path("./logs")
PORT = 8899
HTML_PATH = Path(__file__).parent / "viewer.html"


class EventCache:
    """Thread-safe cache of events from events.jsonl, with file watching."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._last_size = 0
        self._last_sig: tuple[int, int, int] | None = None
        self._last_raw: bytes | None = None
        self._generation = 0
        self._reload()

    def _reload(self):
        """Reload events from file and bump generation on non-append rewrites."""
        events_file = self.log_dir / "events.jsonl"
        if not events_file.exists():
            return
        stat = events_file.stat()
        current_size = stat.st_size
        current_sig = (stat.st_ino, current_size, stat.st_mtime_ns)

        raw = events_file.read_bytes()
        with self._lock:
            if self._last_raw == raw and self._last_sig == current_sig:
                return
            if self._last_raw is not None:
                inode_changed = self._last_sig is not None and stat.st_ino != self._last_sig[0]
                rewritten = inode_changed or not raw.startswith(self._last_raw)
                if rewritten:
                    self._generation += 1

            new_events = []
            for line in raw.decode(errors="replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        new_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            self.events = new_events
            self._last_raw = raw
            self._last_size = current_size
            self._last_sig = current_sig

    def get_since(self, offset: int, client_generation: int | None = None) -> tuple[list[dict], int, int, bool]:
        """Return events since offset and total count."""
        self._reload()
        with self._lock:
            reset = False
            if client_generation is not None and client_generation != self._generation:
                offset = 0
                reset = True
            elif offset > len(self.events):
                offset = 0
                reset = True
            return self.events[offset:], len(self.events), self._generation, reset

    def meta(self) -> dict:
        self._reload()
        events_file = self.log_dir / "events.jsonl"
        with self._lock:
            return {
                "log_dir": str(self.log_dir),
                "events_file": str(events_file),
                "events": len(self.events),
                "generation": self._generation,
                "exists": events_file.exists(),
            }


_cache: EventCache | None = None


def get_cache() -> EventCache:
    global _cache
    if _cache is None:
        _cache = EventCache(LOG_DIR)
    return _cache


def _safe_json(obj):
    """Serialize to JSON, replacing Infinity/NaN with null (valid JSON)."""
    def _default(o):
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return None
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=_default, allow_nan=False)


def _query_int(query: str, name: str) -> int | None:
    for part in query.split("&"):
        key, _, value = part.partition("=")
        if key != name or not value:
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _query_str(query: str, name: str) -> str:
    for part in query.split("&"):
        key, _, value = part.partition("=")
        if key == name:
            return unquote(value)
    return ""


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        raw_path = self.path
        path = raw_path.split("?")[0]
        query = raw_path.split("?")[1] if "?" in raw_path else ""

        if "api/stream" in path:
            self._serve_sse(query)
        elif "api/events" in path:
            self._serve_events(query)
        elif "api/meta" in path:
            self._serve_meta()
        elif "api/llm-logs" in path:
            self._serve_llm_list()
        elif "api/llm-log" in path:
            self._serve_llm_log(query)
        else:
            self._serve_html()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PATH.read_bytes())

    def _serve_sse(self, query):
        """Server-Sent Events endpoint — streams new events in real-time."""
        offset = _query_int(query, "offset") or 0
        client_generation = _query_int(query, "generation")

        # Support Last-Event-ID for auto-reconnect
        last_id = self.headers.get("Last-Event-ID")
        if last_id is not None:
            try:
                offset = int(last_id)
            except ValueError:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.flush()

        cache = get_cache()
        BATCH_SIZE = 30  # send at most 30 events per SSE message
        try:
            while True:
                new_events, total, generation, reset = cache.get_since(offset, client_generation)
                if reset:
                    offset = 0
                    client_generation = generation
                if new_events:
                    # Send in batches to avoid giant single messages
                    first_batch = True
                    while new_events:
                        batch = new_events[:BATCH_SIZE]
                        new_events = new_events[BATCH_SIZE:]
                        offset += len(batch)
                        data = _safe_json({
                            "events": batch,
                            "total": total,
                            "offset": offset,
                            "generation": generation,
                            "reset": reset and first_batch,
                        })
                        first_batch = False
                        # SSE spec: split on newlines, prefix each line with "data:"
                        lines = data.split("\n")
                        sse_data = "\n".join(f"data: {line}" for line in lines)
                        self.wfile.write(f"id: {offset}\n{sse_data}\n\n".encode())
                        self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected

    def _serve_events(self, query):
        """Polling fallback — returns events since offset."""
        offset = _query_int(query, "offset") or 0
        client_generation = _query_int(query, "generation")
        cache = get_cache()
        new_events, total, generation, reset = cache.get_since(offset, client_generation)
        response_offset = len(new_events) if reset else offset + len(new_events)
        self._json_response({
            "events": new_events,
            "total": total,
            "offset": response_offset,
            "generation": generation,
            "reset": reset,
        })

    def _serve_meta(self):
        self._json_response(get_cache().meta())

    def _serve_llm_list(self):
        logs = []
        for f in sorted(LOG_DIR.glob("*.jsonl")):
            if f.name == "events.jsonl":
                continue
            logs.append({"file": f.name, "size": f.stat().st_size})
        self._json_response({"logs": logs, "count": len(logs)})

    def _serve_llm_log(self, query):
        filename = _query_str(query, "file")
        if not filename or ".." in filename or "/" in filename:
            self.send_response(400)
            self.end_headers()
            return
        filepath = LOG_DIR / filename
        if not filepath.exists():
            self.send_response(404)
            self.end_headers()
            return
        try:
            data = json.loads(filepath.read_text())
            if "request" in data and "headers" in data["request"]:
                del data["request"]["headers"]
            self._json_response(data)
        except Exception as e:
            self._json_response({"error": str(e)})

    def _json_response(self, data):
        body = _safe_json(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(http.server.HTTPServer):
    """Handle each request in a new thread (needed for SSE long-polling)."""
    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread, args=(request, client_address))
        t.daemon = True
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    global LOG_DIR, PORT, _cache
    LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./logs")
    PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8899
    _cache = None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"NanoMA Trace Viewer")
    print(f"  Log dir:  {LOG_DIR}")
    print(f"  Events:   {LOG_DIR / 'events.jsonl'}")
    print(f"  HTML:     {HTML_PATH}")
    print(f"  Serving:  http://localhost:{PORT}")
    server = ThreadedHTTPServer(("0.0.0.0", PORT), ViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

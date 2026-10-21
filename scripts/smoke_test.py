"""Smoke test a running deployment (local, Docker, or Render). Standard library only.

    python scripts/smoke_test.py https://medical-intent-classifier.onrender.com
    python scripts/smoke_test.py http://localhost:8000 --requests 50

Waits for a cold start (Render free services sleep after 15 minutes idle), then checks every
endpoint, input validation, the low-confidence fallback, the UI, and request latency.
Exits non-zero if any check fails.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Expected behaviour of the published model (revision f5c6548..., threshold 0.98, T = 1.849).
EXPECTED_THRESHOLD = 0.98
CASES = [
    ("I have a fever", "fever"),
    ("thank you so much", "thanks"),
    ("Hola", "unsure"),             # unknown word -> low confidence
    ("Is this serious?", "unsure"),  # out of scope -> fallback
]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(base, path, body=None, timeout=60, follow=True):
    """Return (status, headers, parsed JSON or text, seconds)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    start = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as resp:
            status, headers, raw = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        status, headers, raw = e.code, e.headers, e.read()
    elapsed = time.perf_counter() - start
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = raw.decode(errors="replace")
    return status, headers, payload, elapsed


def wait_until_ready(base, max_wait):
    start = time.perf_counter()
    attempt = 0
    while time.perf_counter() - start < max_wait:
        attempt += 1
        try:
            status, _, body, _ = request(base, "/health", timeout=30)
            if status == 200 and isinstance(body, dict) and body.get("model_loaded"):
                return time.perf_counter() - start
            note = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            note = type(e).__name__
        print(f"  waiting for service ({note}, attempt {attempt}, {time.perf_counter() - start:.0f}s)")
        time.sleep(5)
    return None


def main():
    parser = argparse.ArgumentParser(description="Smoke test a deployed Medical Intent Classifier.")
    parser.add_argument("base_url", help="e.g. https://medical-intent-classifier.onrender.com")
    parser.add_argument("--requests", type=int, default=30, help="requests for the latency measurement")
    parser.add_argument("--max-wait", type=int, default=300, help="seconds to wait for a cold start")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    for suffix in ("/ui", "/docs"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]

    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    print(f"Target: {base}")
    ready_after = wait_until_ready(base, args.max_wait)
    check("service becomes ready", ready_after is not None,
          f"ready after {ready_after:.1f}s" if ready_after is not None else f"not ready after {args.max_wait}s")
    if ready_after is None:
        sys.exit(1)

    status, _, info, _ = request(base, "/model-info")
    check("GET /model-info", status == 200 and info.get("backend") == "onnx" and len(info.get("labels", [])) == 12,
          f"backend={info.get('backend')}, labels={len(info.get('labels', []))}")
    check("validated threshold in use", info.get("confidence_threshold") == EXPECTED_THRESHOLD,
          f"threshold={info.get('confidence_threshold')}, temperature={info.get('temperature')}")
    check("disclaimer present", "medical advice" in info.get("disclaimer", ""))

    for text, expected in CASES:
        status, _, body, secs = request(base, "/predict", {"text": text})
        got = body.get("intent") if isinstance(body, dict) else None
        conf = body.get("confidence", 0) if isinstance(body, dict) else 0
        check(f"POST /predict {text!r} -> {expected}", status == 200 and got == expected,
              f"got {got} @ {conf:.3f}, {secs * 1000:.0f} ms")

    status, _, body, _ = request(base, "/predict/batch", {"texts": ["hello", "my head hurts a lot"]})
    results = body.get("results", []) if isinstance(body, dict) else []
    check("POST /predict/batch", status == 200 and len(results) == 2,
          ", ".join(f"{r['text']!r}->{r['intent']}" for r in results))

    for name, payload in [("empty text", {"text": ""}), ("text over 500 chars", {"text": "a" * 501}),
                          ("missing field", {})]:
        status, _, _, _ = request(base, "/predict", payload)
        check(f"rejects {name} with 422", status == 422, f"HTTP {status}")

    status, headers, _, _ = request(base, "/", follow=False)
    check("GET / redirects to the UI", status in (301, 302, 307, 308) and "/ui" in headers.get("location", ""),
          f"HTTP {status} -> {headers.get('location')}")
    status, _, page, secs = request(base, "/ui/")
    check("GET /ui/ serves the Gradio app", status == 200 and "gradio" in str(page).lower(),
          f"HTTP {status}, {secs:.2f}s")
    status, _, _, _ = request(base, "/docs")
    check("GET /docs", status == 200, f"HTTP {status}")

    times = []
    for i in range(args.requests):
        status, _, _, secs = request(base, "/predict", {"text": CASES[i % len(CASES)][0]})
        if status == 200:
            times.append(secs * 1000)
    if times:
        times.sort()
        p95 = times[min(len(times) - 1, int(round(0.95 * len(times))) - 1)]
        print(f"\nLatency over HTTPS, {len(times)} sequential /predict requests (includes network round trip):")
        p50 = statistics.median(times)
        print(f"  p50 {p50:.0f} ms   p95 {p95:.0f} ms   min {times[0]:.0f} ms   max {times[-1]:.0f} ms")
    check("latency requests succeeded", len(times) == args.requests, f"{len(times)}/{args.requests} returned 200")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

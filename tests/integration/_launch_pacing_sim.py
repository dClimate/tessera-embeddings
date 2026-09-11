"""A local stand-in for EC2 that enforces the account's RunInstances request quota.

The quota is a token bucket — a small burst capacity refilled at a fixed rate — shared
by every caller in the account and not adjustable. This module reproduces it, speaks
just enough of the EC2 query protocol for boto3 to parse a launch or raise
``RequestLimitExceeded``, and drives **Ray's own AWS node provider** at it. Only the
endpoint is a stand-in: the client, its retry configuration, and the subnet-rotating
retry loop around the launch are all Ray's.

Run as a script, in two modes, because the topology matters. Each cluster's autoscaler
is its own process on its own head node, so each has its own botocore client and
therefore its own client-side rate limiter. Ray caches the EC2 resource per
(region, max_retries), so several clusters inside one process would share ONE limiter —
a coordination they do not have in the field.

    python _launch_pacing_sim.py serve     # own the shared bucket, report on /stats
    python _launch_pacing_sim.py cluster   # drive ONE cluster's launch dispatch

Environment read in ``cluster`` mode: ``SIM_PORT``, ``SIM_NODES``, ``SIM_INDEX``.
In ``serve`` mode: ``SIM_BUCKET_CLOSED`` ("1" holds the bucket shut until ``/open``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

#: The account's RunInstances request bucket. Not adjustable, so the same in every account.
BUCKET_CAPACITY = 5.0
BUCKET_REFILL_PER_SEC = 2.0

#: Service-side latency, so the offered load is shaped like an API round trip rather
#: than by loopback speed. A refused request comes back fast; a launch that is actually
#: placed does not.
LATENCY_THROTTLED_S = 0.05
LATENCY_ACCEPTED_S = 0.40

REGION = "us-west-2"
SUBNETS = ["subnet-aaa", "subnet-bbb", "subnet-ccc"]

_LAUNCHED = """<?xml version="1.0" encoding="UTF-8"?>
<RunInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">
  <requestId>req-{seq}</requestId>
  <reservationId>r-{seq}</reservationId>
  <ownerId>123456789012</ownerId>
  <groupSet/>
  <instancesSet>{items}</instancesSet>
</RunInstancesResponse>
"""

_ITEM = """
    <item>
      <instanceId>i-{iid:017x}</instanceId>
      <imageId>ami-0123456789abcdef0</imageId>
      <instanceState><code>0</code><name>pending</name></instanceState>
      <stateReason><code>pending</code><message>pending</message></stateReason>
      <instanceType>g6e.xlarge</instanceType>
      <placement><availabilityZone>us-west-2a</availabilityZone></placement>
    </item>"""

_THROTTLED = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Errors>
    <Error>
      <Code>RequestLimitExceeded</Code>
      <Message>Request limit exceeded.</Message>
    </Error>
  </Errors>
  <RequestID>req-throttled</RequestID>
</Response>
"""

_UNSUPPORTED = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Errors><Error><Code>InvalidAction</Code><Message>{action}</Message></Error></Errors>
  <RequestID>req-unsupported</RequestID>
</Response>
"""


class RequestBucket:
    """The shared request bucket, plus a log of every call it answered."""

    def __init__(self, *, closed: bool = False) -> None:
        self._tokens = BUCKET_CAPACITY
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.closed = closed
        self.calls: list[tuple[float, str, int]] = []
        self._t0 = time.monotonic()
        self._seq = 0

    def take(self, max_count: int) -> tuple[bool, int]:
        """Answer one RunInstances call; return (granted, sequence number)."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(BUCKET_CAPACITY, self._tokens + (now - self._last) * BUCKET_REFILL_PER_SEC)
            self._last = now
            granted = not self.closed and self._tokens >= 1.0
            if granted:
                self._tokens -= 1.0
            self._seq += 1
            self.calls.append((now - self._t0, "accepted" if granted else "throttled", max_count))
            return granted, self._seq

    def stats(self) -> dict[str, object]:
        with self._lock:
            accepted = [c for c in self.calls if c[1] == "accepted"]
            throttled = [c for c in self.calls if c[1] == "throttled"]
            span = self.calls[-1][0] - self.calls[0][0] if len(self.calls) > 1 else 0.0
            return {
                "calls": len(self.calls),
                "accepted": len(accepted),
                "throttled": len(throttled),
                "span_s": round(span, 3),
                "log": [[round(t, 4), outcome] for t, outcome, _ in self.calls],
            }


def serve(bucket: RequestBucket) -> ThreadingHTTPServer:
    """Start the stand-in endpoint on an ephemeral port; the caller shuts it down."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return  # keep the test output readable

        def _reply(self, status: int, body: str) -> None:
            payload = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/xml")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/stats":
                self._reply(200, json.dumps(bucket.stats()))
            elif self.path == "/open":
                bucket.closed = False
                self._reply(200, "<ok/>")
            else:
                self._reply(404, "<no/>")

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            form = {k: v[0] for k, v in parse_qs(raw).items()}
            action = form.get("Action", "")
            if action != "RunInstances":
                self._reply(400, _UNSUPPORTED.format(action=action))
                return
            max_count = int(form.get("MaxCount", "1"))
            granted, seq = bucket.take(max_count)
            if not granted:
                time.sleep(LATENCY_THROTTLED_S)
                self._reply(503, _THROTTLED)
                return
            time.sleep(LATENCY_ACCEPTED_S)
            items = "".join(_ITEM.format(iid=seq * 1000 + i) for i in range(max_count))
            self._reply(200, _LAUNCHED.format(seq=seq, items=items))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _serve_mode() -> None:
    bucket = RequestBucket(closed=os.environ.get("SIM_BUCKET_CLOSED") == "1")
    server = serve(bucket)
    # stdout is this process's only channel to the orchestrator, which reads the
    # port from the first line.
    print(json.dumps({"port": server.server_address[1]}), flush=True)  # noqa: T201
    sys.stdin.read()  # held open until the orchestrator closes stdin
    server.shutdown()


def _cluster_mode() -> None:
    os.environ["AWS_ENDPOINT_URL_EC2"] = f"http://127.0.0.1:{os.environ['SIM_PORT']}"
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", REGION)
    logging.disable(logging.WARNING)

    # Imported only now. The autoscaler constants are module-level environment reads,
    # and botocore resolves the retry mode when the client is built — so both have to
    # see this process's environment, not the orchestrator's.
    from ray.autoscaler._private.aws.node_provider import AWSNodeProvider
    from ray.autoscaler._private.constants import (
        AUTOSCALER_MAX_CONCURRENT_LAUNCHES,
        AUTOSCALER_MAX_LAUNCH_BATCH,
    )

    node_config = {
        "InstanceType": "g6e.xlarge",
        "ImageId": "ami-0123456789abcdef0",
        "KeyName": "k",
        "SecurityGroupIds": ["sg-1"],
        "SubnetIds": list(SUBNETS),
    }
    tags = {"ray-node-kind": "worker", "ray-launch-config": "h", "ray-node-name": "w"}

    provider = AWSNodeProvider(
        {"region": REGION, "cache_stopped_nodes": False}, f"sim-{os.environ.get('SIM_INDEX', '0')}"
    )

    # `StandardAutoscaler.launch_new_node`: the requested count is split into
    # batch-sized calls and drained by a fixed pool of launcher threads.
    work: queue.Queue[int] = queue.Queue()
    remaining = int(os.environ.get("SIM_NODES", "75"))
    while remaining > 0:
        work.put(min(remaining, AUTOSCALER_MAX_LAUNCH_BATCH))
        remaining -= AUTOSCALER_MAX_LAUNCH_BATCH

    placed = [0]
    lock = threading.Lock()

    def launcher() -> None:
        while True:
            try:
                count = work.get_nowait()
            except queue.Empty:
                return
            try:
                created = provider.create_node(dict(node_config), dict(tags), count)
            except Exception:
                continue
            with lock:
                placed[0] += len(created)

    threads = [
        threading.Thread(target=launcher)
        for _ in range(max(1, math.ceil(AUTOSCALER_MAX_CONCURRENT_LAUNCHES / AUTOSCALER_MAX_LAUNCH_BATCH)))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(json.dumps({"placed": placed[0]}), flush=True)  # noqa: T201 - the result channel


if __name__ == "__main__":
    {"serve": _serve_mode, "cluster": _cluster_mode}[sys.argv[1]]()

"""The library stacks the tests stage their tiles under (``ZarrWriter.write_chunk(inference_stack=...)``).

``STACK`` for every test that stages a tile, so a store its tiles stamped accepts more of them;
``OTHER_STACK`` for a test about a mismatch.
"""

STACK = {"torch": "2.5.1+cu121", "cuda": "12.1"}
OTHER_STACK = {"torch": "2.14.1+cu128", "cuda": "12.8"}

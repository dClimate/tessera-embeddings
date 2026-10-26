"""The library stacks and models the tests stage their tiles under (``ZarrWriter.write_chunk``).

``STACK`` and ``MODEL`` for every test that stages a tile, so a store its tiles stamped (or a
store seeded with the default model) accepts more of them; ``OTHER_*`` for a test about a mismatch.
"""

STACK = {"torch": "2.5.1+cu121", "cuda": "12.1"}
OTHER_STACK = {"torch": "2.14.1+cu128", "cuda": "12.8"}
MODEL = {"model_version": "v1.1", "checkpoint_id": "tessera_v1_1_aws_encoder"}
OTHER_MODEL = {"model_version": "v2-large", "checkpoint_id": "v2_student_large"}

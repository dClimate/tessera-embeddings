"""The stacks the tests stage their runs under (``ZarrWriter.write_chunk(stack=...)``).

``STACK`` for every test that stages a tile, so a store its runs stamped accepts more of them;
``OTHER_STACK`` (another PyTorch) for a test about a mismatch.
"""

STACK = {"model_version": "v1.1", "checkpoint_id": "tessera_v1_1_aws_encoder", "torch": "2.5.1+cu121", "cuda": "12.1"}
OTHER_STACK = {**STACK, "torch": "2.14.1+cu128", "cuda": "12.8"}


def identity(stack: dict) -> dict:
    """What a caller labels a store with for a run staged on *stack*: ``assemble``'s model and checkpoint."""
    return {"model_version": stack["model_version"], "checkpoint_id": stack["checkpoint_id"]}


IDENTITY = identity(STACK)

from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if old in text:
        text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")
        print(f"[PATCHED] {path}")
        return

    if new in text:
        print(f"[ALREADY PATCHED] {path}")
        return

    raise RuntimeError(f"Could not find expected text in {path}")


# ============================================================
# 1. Increase max_tokens: 50 -> 128
# ============================================================

replace_once(
    "src/api_client.py",
    "    max_tokens: int = 50,",
    "    max_tokens: int = 128,"
)


# ============================================================
# 2. Retry once when response is invalid/incomplete/out-of-label
# ============================================================

old_api_end = """    if response is None:
        return None

    return extract_prediction_from_response(response, logger)
"""

new_api_end = """    if response is None:
        return None

    prediction = extract_prediction_from_response(response, logger)

    # Accept only labels that belong to the current dataset.
    if prediction in config.labels:
        return prediction

    if prediction is None:
        logger.warning(
            "Invalid or incomplete model response. Retrying once with the same prompt."
        )
    else:
        logger.warning(
            "Prediction %r is outside allowed labels %s. Retrying once.",
            prediction,
            config.labels,
        )

    time.sleep(1.0)

    response = send_api_request(
        url=config.api_url,
        headers=headers,
        payload=payload,
        max_retries=config.max_retries,
        initial_delay=config.initial_retry_delay,
        logger=logger,
    )

    if response is None:
        return None

    prediction = extract_prediction_from_response(response, logger)

    if prediction in config.labels:
        return prediction

    if prediction is not None:
        logger.warning(
            "Retry prediction %r is still outside allowed labels %s.",
            prediction,
            config.labels,
        )

    return None
"""

replace_once(
    "src/api_client.py",
    old_api_end,
    new_api_end
)


# ============================================================
# 3. Preserve Half-True as Half-True
# ============================================================

label_anchor = """    if clean_s in {
        "not enough information","""

half_true_block = """    # Preserve Half-True for datasets such as TRACER and Fact5.
    if clean_s in {
        "half-true",
        "half true",
        "halftrue",
        "partlytrue",
        "partiallytrue",
        "partially true",
    }:
        return "Half-True"

    if clean_s in {
        "not enough information","""

replace_once(
    "src/label_utils.py",
    label_anchor,
    half_true_block
)


# ============================================================
# 4. Add chunking options to Config
# ============================================================

replace_once(
    "src/config.py",
    """    output_root: Path
    api_key: str = """"",
    """    output_root: Path
    chunk_index: Optional[int] = None
    num_chunks: Optional[int] = None
    api_key: str = """""
)

replace_once(
    "src/config.py",
    """        samp = cfg.get("sampling", {})
        prompt_methods = cfg.get("prompt_methods", ["naive"])""",
    """        samp = cfg.get("sampling", {})
        chunk = cfg.get("chunking", {})
        prompt_methods = cfg.get("prompt_methods", ["naive"])"""
)

replace_once(
    "src/config.py",
    """            output_root=Path(cfg.get("output_root", "results")),
            api_key=api_key,""",
    """            output_root=Path(cfg.get("output_root", "results")),
            chunk_index=chunk.get("index"),
            num_chunks=chunk.get("num_chunks"),
            api_key=api_key,"""
)


# ============================================================
# 5. Chunk seen and unseen separately
# ============================================================

pipeline_anchor = """        # 2. If balanced sampling is enabled, sample BEFORE inference."""

pipeline_chunk_block = """        # Optional chunking for long-running jobs.
        # Seen and unseen are chunked separately so every chunk contains
        # approximately the same fraction of both temporal splits.
        if self.config.num_chunks is not None:
            if self.config.chunk_index is None:
                raise ValueError(
                    "chunking.index must be provided when chunking.num_chunks is set"
                )

            chunk_index = self.config.chunk_index
            num_chunks = self.config.num_chunks

            if num_chunks <= 0:
                raise ValueError("chunking.num_chunks must be greater than zero")

            if not 0 <= chunk_index < num_chunks:
                raise ValueError(
                    f"chunking.index must be between 0 and {num_chunks - 1}"
                )

            def slice_chunk(items):
                start = len(items) * chunk_index // num_chunks
                end = len(items) * (chunk_index + 1) // num_chunks
                return items[start:end]

            original_seen = len(seen_claims)
            original_unseen = len(unseen_claims)

            seen_claims = slice_chunk(seen_claims)
            unseen_claims = slice_chunk(unseen_claims)

            print(
                f"🧩 Chunk {chunk_index + 1}/{num_chunks}: "
                f"seen {len(seen_claims)}/{original_seen}, "
                f"unseen {len(unseen_claims)}/{original_unseen}"
            )

        # 2. If balanced sampling is enabled, sample BEFORE inference."""

replace_once(
    "src/pipeline.py",
    pipeline_anchor,
    pipeline_chunk_block
)

print()
print("All patches applied successfully.")
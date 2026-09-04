from pathlib import Path


def read(path):
    return Path(path).read_text(encoding="utf-8")


def write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def insert_after_line(path, contains, new_lines, marker):
    text = read(path)

    if marker in text:
        print(f"[ALREADY PATCHED] {path}: {marker}")
        return

    lines = text.splitlines()

    for i, line in enumerate(lines):
        if contains in line:
            lines[i + 1:i + 1] = new_lines
            write(path, "\n".join(lines) + "\n")
            print(f"[PATCHED] {path}: {marker}")
            return

    raise RuntimeError(
        f"Could not find line containing {contains!r} in {path}"
    )


# ============================================================
# 1. API client
# ============================================================

path = "src/api_client.py"
text = read(path)

# max_tokens may already have been patched by the previous script.
if "max_tokens: int = 50," in text:
    text = text.replace(
        "max_tokens: int = 50,",
        "max_tokens: int = 128,",
        1,
    )
    print("[PATCHED] api_client max_tokens -> 128")
elif "max_tokens: int = 128," in text:
    print("[ALREADY PATCHED] api_client max_tokens = 128")
else:
    raise RuntimeError("Could not find max_tokens in api_client.py")

retry_marker = "Invalid or incomplete model response. Retrying once"

if retry_marker not in text:

    target = "    return extract_prediction_from_response(response, logger)"

    pos = text.rfind(target)

    if pos == -1:
        raise RuntimeError(
            "Could not find final extract_prediction_from_response call "
            "in src/api_client.py"
        )

    replacement = '''    prediction = extract_prediction_from_response(
        response,
        logger,
    )

    # Only accept labels valid for the current dataset.
    if prediction in config.labels:
        return prediction

    if prediction is None:
        logger.warning(
            "Invalid or incomplete model response. Retrying once "
            "with the same prompt."
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

    prediction = extract_prediction_from_response(
        response,
        logger,
    )

    if prediction in config.labels:
        return prediction

    if prediction is not None:
        logger.warning(
            "Retry prediction %r is outside allowed labels %s.",
            prediction,
            config.labels,
        )

    return None'''

    text = text[:pos] + replacement + text[pos + len(target):]

    print("[PATCHED] api_client invalid-output retry")
else:
    print("[ALREADY PATCHED] api_client retry")

write(path, text)


# ============================================================
# 2. Preserve Half-True
# ============================================================

path = "src/label_utils.py"
text = read(path)

marker = "# Preserve Half-True for three/four-class datasets."

if marker not in text:
    lines = text.splitlines()

    inserted = False

    for i, line in enumerate(lines):
        if "clean_s = re.sub" in line:

            block = [
                "",
                f"    {marker}",
                "    if clean_s in {",
                '        "half-true",',
                '        "half true",',
                '        "halftrue",',
                "    }:",
                '        return "Half-True"',
            ]

            lines[i + 1:i + 1] = block
            inserted = True
            break

    if not inserted:
        raise RuntimeError(
            "Could not find clean_s line in src/label_utils.py"
        )

    write(path, "\n".join(lines) + "\n")
    print("[PATCHED] label_utils Half-True")
else:
    print("[ALREADY PATCHED] label_utils Half-True")


# ============================================================
# 3. Config: add chunk settings
# ============================================================

path = "src/config.py"

insert_after_line(
    path,
    "output_root: Path",
    [
        "    chunk_index: Optional[int] = None",
        "    num_chunks: Optional[int] = None",
    ],
    "chunk_index: Optional[int]",
)

insert_after_line(
    path,
    'samp = cfg.get("sampling", {})',
    [
        '        chunk = cfg.get("chunking", {})',
    ],
    'chunk = cfg.get("chunking", {})',
)

insert_after_line(
    path,
    'output_root=Path(cfg.get("output_root", "results"))',
    [
        '            chunk_index=chunk.get("index"),',
        '            num_chunks=chunk.get("num_chunks"),',
    ],
    'chunk_index=chunk.get("index")',
)


# ============================================================
# 4. Pipeline: chunk seen/unseen separately
# ============================================================

path = "src/pipeline.py"
text = read(path)

chunk_marker = "# OPTIONAL CHUNKED INFERENCE"

if chunk_marker not in text:

    anchor = (
        "        # 2. If balanced sampling is enabled, "
        "sample BEFORE inference."
    )

    if anchor not in text:
        raise RuntimeError(
            "Could not find sampling anchor in src/pipeline.py"
        )

    block = '''        # OPTIONAL CHUNKED INFERENCE
        if self.config.num_chunks is not None:

            if self.config.chunk_index is None:
                raise ValueError(
                    "chunking.index must be provided when "
                    "chunking.num_chunks is set"
                )

            chunk_index = self.config.chunk_index
            num_chunks = self.config.num_chunks

            if num_chunks <= 0:
                raise ValueError(
                    "chunking.num_chunks must be greater than zero"
                )

            if not 0 <= chunk_index < num_chunks:
                raise ValueError(
                    f"chunking.index must be between "
                    f"0 and {num_chunks - 1}"
                )

            def slice_chunk(items):
                start = (
                    len(items) * chunk_index // num_chunks
                )
                end = (
                    len(items) * (chunk_index + 1) // num_chunks
                )
                return items[start:end]

            original_seen = len(seen_claims)
            original_unseen = len(unseen_claims)

            seen_claims = slice_chunk(seen_claims)
            unseen_claims = slice_chunk(unseen_claims)

            print(
                f"Chunk {chunk_index + 1}/{num_chunks}: "
                f"seen {len(seen_claims)}/{original_seen}, "
                f"unseen {len(unseen_claims)}/{original_unseen}"
            )

'''

    text = text.replace(anchor, block + anchor, 1)
    write(path, text)

    print("[PATCHED] pipeline chunking")
else:
    print("[ALREADY PATCHED] pipeline chunking")


print()
print("============================================")
print("ALL PATCHES APPLIED SUCCESSFULLY")
print("============================================")
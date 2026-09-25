"""Remove table padding introduced by embedded images in document exports."""

import re
import uuid

from logger import logger


def _table_cells(line):
    if line.startswith(("    ", "\t")):
        return None
    # Jump between Markdown punctuation rather than walking huge padding runs.
    tokens = list(re.finditer(r"\\.|`+|\|", line))
    next_tick = {}
    closing_ticks = {}
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index].group()
        if token.startswith("`"):
            closing_ticks[index] = next_tick.get(len(token))
            next_tick[len(token)] = index
    pipes = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.group().startswith("`") and closing_ticks[index] is not None:
            index = closing_ticks[index] + 1
            continue
        if token.group() == "|":
            pipes.append(token.start())
        index += 1
    if not pipes:
        return None
    cells = []
    start = 0
    for position in pipes:
        cells.append(line[start:position].strip(" \t"))
        start = position + 1
    cells.append(line[start:].strip(" \t"))
    if not cells[0]:
        cells.pop(0)
    if not cells[-1]:
        cells.pop()
    return cells or None


def normalize_document_markdown(markdown, *, source="document"):
    """Compact recognized pipe tables without changing cell contents or code."""
    output = []
    previous_cells = None
    table_width = None
    fence = None
    for raw in markdown.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        ending = raw[len(line):]
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            output.append(raw)
            if (fence_match and fence_match[1][0] == fence[0]
                    and len(fence_match[1]) >= len(fence)
                    and not fence_match[2].strip()):
                fence = None
            continue
        if fence_match and not (
                fence_match[1][0] == "`" and "`" in fence_match[2]):
            fence = fence_match[1]
            previous_cells = None
            table_width = None
            output.append(raw)
            continue
        cells = _table_cells(line)
        separator = bool(cells) and all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in cells)
        if (separator and previous_cells is not None
                and len(cells) == len(previous_cells)):
            previous = output[-1]
            previous_ending = previous[len(previous.rstrip("\r\n")):]
            output[-1] = "| " + " | ".join(previous_cells) + " |" + previous_ending
            cells = [(":" if cell.startswith(":") else "") + "---"
                     + (":" if cell.endswith(":") else "") for cell in cells]
            table_width = len(cells)
            output.append("| " + " | ".join(cells) + " |" + ending)
        elif table_width is not None and cells and len(cells) == table_width:
            output.append("| " + " | ".join(cells) + " |" + ending)
        else:
            table_width = None
            output.append(raw)
        previous_cells = cells
    result = "".join(output)
    if result != markdown:
        logger.info(
            "Markdown table normalization source=%s before_bytes=%s after_bytes=%s",
            source, len(markdown.encode("utf-8")), len(result.encode("utf-8")))
    return result


def normalize_history_documents(message):
    """Normalize bounded document blocks in memory, preserving image objects."""
    if message.get("role") != "user" or not isinstance(message.get("content"), list):
        return message
    blocks = message["content"]
    output = []
    index = 0
    while index < len(blocks):
        first = blocks[index]
        text = first.get("text", "") if first.get("type") == "text" else ""
        begin = re.match(r'^# FILE "([^\r\n]+)" BEGIN\n', text)
        if not begin:
            if "# FILE " in text and (" BEGIN" in text or " END" in text):
                logger.warning("History document normalization skipped: ambiguous boundary")
            output.append(first)
            index += 1
            continue
        end_marker = '\n# FILE "' + begin[1] + '" END'
        end = index
        parts = []
        images = []
        marker_prefix = "DOCIMAGE" + uuid.uuid4().hex + "X"
        valid = False
        while end < len(blocks):
            block = blocks[end]
            if block.get("type") == "text" and set(block) == {"type", "text"}:
                value = block["text"]
                if end != index and re.search(r'^# FILE .* BEGIN', value, re.MULTILINE):
                    break
                parts.append(value)
                if end_marker in value:
                    valid = value.endswith(end_marker) and value.count(end_marker) == 1
                    break
            elif block.get("type") == "image_url":
                token = marker_prefix + str(len(images)) + "END"
                images.append((token, block))
                parts.append(token)
            else:
                break
            end += 1
        if not valid or any(marker_prefix in block.get("text", "") for block in blocks):
            logger.warning("History document normalization skipped: ambiguous boundary")
            output.append(first)
            index += 1
            continue
        original = "".join(parts)
        boundaries = re.findall(r'^# FILE "[^\r\n]+" (BEGIN|END)$', original, re.MULTILINE)
        if boundaries != ["BEGIN", "END"]:
            logger.warning("History document normalization skipped: ambiguous boundary")
            output.extend(blocks[index:end + 1])
            index = end + 1
            continue
        normalized = normalize_document_markdown(original, source="history")
        if normalized == original:
            output.extend(blocks[index:end + 1])
        else:
            start = 0
            for token, image in images:
                position = normalized.index(token, start)
                if position > start:
                    output.append({"type": "text", "text": normalized[start:position]})
                output.append(image)
                start = position + len(token)
            if start < len(normalized):
                output.append({"type": "text", "text": normalized[start:]})
        index = end + 1
    return {**message, "content": output}

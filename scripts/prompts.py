"""Prompt template(s) for the full-image reading task.

Kept in its own module so the exact wording used for a scored run is
versioned and citable, rather than inlined in the runner script.
"""

READ_ALL_TEXT = (
    "You are looking at a real, unedited photograph taken somewhere in India. "
    "Read every piece of legible text visible in the image: shop signs, "
    "boards, banners, posters, hoardings, labels, name plates, price lists, "
    "anything with writing on it.\n\n"
    "Transcribe each distinct piece of text exactly as it is written, in its "
    "original script. Do not translate or transliterate. Do not describe the "
    "image or the objects in it. List one piece of text per line, in the "
    "order you notice them. If a piece of text is present but not legible, "
    "skip it rather than guessing.\n\n"
    "Output only the transcribed lines, nothing else."
)

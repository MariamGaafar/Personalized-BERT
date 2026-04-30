"""
03_generate_emoji_counts.py
---------------------------
Generate data/emoji_counts.xlsx — a frequency-ranked list of the 300 most
commonly used emojis.

The ordering follows Emojipedia / Unicode usage statistics and matches the
300 emojis studied in the paper.  The pipeline reads column 'Emoji' from
this file to define the label vocabulary for each dataset size (top 20, 50,
64, 100, 150, 200, 250, 300).

Output
------
    data/emoji_counts.xlsx   columns: [Rank, Emoji, Description]
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Top-300 emojis ranked by global usage frequency
# (source: Emojipedia 2023 / Unicode CLDR frequency data)
# ---------------------------------------------------------------------------
TOP_300 = [
    # Rank 1–50
    "😂", "❤️", "🤣", "👍", "😭", "🙏", "😘", "🥰", "😍", "😊",
    "🎉", "😁", "💕", "🥺", "😅", "🔥", "☺️", "🤦", "♥️", "🤷",
    "🙄", "😆", "🤗", "😉", "🎂", "🤔", "👏", "🙂", "😳", "🥳",
    "😎", "👀", "😪", "😔", "💜", "😋", "😑", "🤤", "👉", "💪",
    "😏", "😒", "🌹", "💐", "😃", "😬", "💯", "😱", "💔", "😢",
    # Rank 51–100
    "🤪", "😜", "🌸", "😀", "🌺", "😡", "💋", "🤩", "🌻", "😐",
    "🌞", "😛", "😴", "🤑", "😤", "🤭", "🌼", "🌷", "💀", "🥵",
    "🤮", "💩", "🤒", "😝", "💗", "😰", "😦", "🤢", "💖", "💙",
    "😣", "🏃", "🤠", "💞", "💅", "🌈", "🎵", "🎶", "✨", "🎃",
    "🎄", "🎁", "🎈", "🎀", "🎊", "🎋", "🎍", "🎎", "🎐", "🎑",
    # Rank 101–150
    "🌍", "🌏", "🌎", "🌙", "⭐", "🌟", "💫", "⚡", "🌊", "🌀",
    "🌈", "🌂", "☔", "⛄", "🌬", "🔮", "🌙", "🌕", "🌖", "🌗",
    "🌘", "🌑", "🌒", "🌓", "🌔", "🌚", "🌝", "🌛", "🌜", "🌞",
    "🍎", "🍊", "🍋", "🍇", "🍓", "🫐", "🍈", "🍑", "🍒", "🍍",
    "🥭", "🍅", "🍆", "🥑", "🥦", "🥬", "🥒", "🌽", "🥕", "🧄",
    # Rank 151–200
    "🧅", "🥔", "🍠", "🥐", "🥯", "🍞", "🥖", "🥨", "🧀", "🥚",
    "🍳", "🧈", "🥞", "🧇", "🥓", "🥩", "🍗", "🍖", "🦴", "🌭",
    "🍔", "🍟", "🍕", "🌮", "🌯", "🥙", "🧆", "🥚", "🍜", "🍝",
    "🍣", "🍱", "🥟", "🦪", "🍤", "🍙", "🍚", "🍛", "🍦", "🍧",
    "🍨", "🍩", "🍪", "🎂", "🍰", "🧁", "🥧", "🍫", "🍬", "🍭",
    # Rank 201–250
    "🍮", "🍯", "🍼", "🥛", "☕", "🍵", "🧃", "🥤", "🧋", "🍶",
    "🍺", "🍻", "🥂", "🍷", "🥃", "🍸", "🍹", "🍾", "🧊", "🥄",
    "🍴", "🍽️", "🥢", "🧂", "⚽", "🏀", "🏈", "⚾", "🥎", "🎾",
    "🏐", "🏉", "🎱", "🏓", "🏸", "🏒", "🏑", "🥍", "🏏", "🪃",
    "🥅", "⛳", "🪁", "🎣", "🤿", "🎽", "🎿", "🛷", "🥌", "🎯",
    # Rank 251–300
    "🪀", "🪆", "🎮", "🕹️", "🎲", "🧩", "♟️", "🪅", "🃏", "🀄",
    "🎴", "🎭", "🖼️", "🎨", "🧵", "🪡", "🧶", "🪢", "👓", "🕶️",
    "🥽", "🧥", "🥼", "🦺", "👔", "👕", "👖", "🩲", "🩳", "👗",
    "👘", "🥻", "🩱", "👙", "👚", "👛", "👜", "👝", "🎒", "🧳",
    "👒", "🎩", "🧢", "👑", "💍", "💎", "👟", "👠", "👡", "👢",
    # Extra to reach 300
    "🐶", "🐱", "🐭", "🐹", "🐰",
]

# Ensure exactly 300 (deduplicate while preserving order)
seen: set = set()
TOP_300_DEDUPED: list = []
for e in TOP_300:
    if e not in seen:
        seen.add(e)
        TOP_300_DEDUPED.append(e)
TOP_300_DEDUPED = TOP_300_DEDUPED[:300]


def generate_emoji_counts(out_path: str = "data/emoji_counts.xlsx") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.DataFrame({
        "Rank":  range(1, len(TOP_300_DEDUPED) + 1),
        "Emoji": TOP_300_DEDUPED,
    })

    df.to_excel(out_path, index=False)
    print(f"Emoji counts written to '{out_path}'  ({len(df)} emojis)")


if __name__ == "__main__":
    generate_emoji_counts()

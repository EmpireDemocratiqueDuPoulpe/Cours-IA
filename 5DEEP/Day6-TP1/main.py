from pathlib import Path
import colorama
from colorama import Fore, Style

# ### Constants ########################################################################################################

# ### File processing ##################################################################################################
def load_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")

# ### Text processing ##################################################################################################
def prepare_text(text: str) -> str:
    return text.lower()

def count_characters(text: str, show: int = -1) -> None:
    # Count occurrences
    occurrences = {}

    for char in set(text):
        occurrences[char.replace("\n","\\n")] = text.count(char)

    sorted_occurrences = sorted(occurrences.items(), key=lambda o: o[1], reverse=True)

    # Print the result
    sorted_occurrences_length = len(sorted_occurrences)
    limit = sorted_occurrences_length if (show == -1) else min(max(0, show), sorted_occurrences_length)

    print(f"{Style.BRIGHT}Ordered list of characters occurrences:")
    for idx in range(limit):
        item = sorted_occurrences[idx]
        print(fr"{idx + 1}.  [ {item[0]} ] - {item[1]} occurrence{'s' if (item[1] > 1) else ''}")

    if limit < sorted_occurrences_length:
        print("  ...")


def main() -> None:
    # Load the book
    book = load_text(filepath=(Path(__file__).resolve().parent / "data" / "voyage-au-centre-de-la-terre_jules-verne_trimmed-version.txt"))

    # Prepare the book
    book = prepare_text(book)
    print(book)

    # Text analysis
    count_characters(book, show=20)


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()

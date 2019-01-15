from src import preprocessing
from src.utils import utils_io


def main():

    root = utils_io.root_folder()
    doc = utils_io.load_raw(f"{root}/data/raw/book_1.txt")
    doc = " ".join(doc)  # Combine into single string

    sents = preprocessing.tokenise_sentences(doc)

    for i in range(len(sents)):

        sents[i] = sents[i].lower()
        sents[i] = preprocessing.remove_nonalpha(sents[i])
        sents[i] = preprocessing.remove_blanks(sents[i])

    # Save cleaned data
    utils_io.save_data(sents, f"{root}/data/interim/book_1.txt")


if __name__ == '__main__':
    main()
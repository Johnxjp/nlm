"""IO functions"""

import os
import pickle
import json


def src_folder():
    return os.path.dirname(os.path.dirname(__file__))


def root_folder():
    return os.path.dirname(src_folder())


def get_full_path(file, base):

    if base == "root":
        return f"{root_folder()}/{file}"
    elif base == "src":
        return f"{src_folder()}/{file}"
    else:
        raise Exception("Invalid base name. Expect 'root' or 'src'.")


def pickle_save(object, file):
    pickle.dump(object, open(file, 'wb'))


def pickle_load(file):
    return pickle.load(open(file, 'rb'))


def json_load(file):
    return json.load(open(file, 'r'))


def json_save(object, file):
    json.dump(object, open(file, 'w'), indent=4)


def load_raw(file):

    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip non-text lines
            if line == "" or "*" in line:
                continue

            data.append(line)

    return data


def save_data(data, file):

    with open(file, 'w') as f:
        for line in data:
            f.write(f"{line}\n")


def load_clean(file):

    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()

            data.append(line)

    return data

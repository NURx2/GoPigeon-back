import pandas as pd

imagenet = None
allowed_species = None


def load_imagenet():
    global imagenet, allowed_species

    print("reading csv")
    imagenet = pd.read_csv("imagenet.csv").fillna("")

    allowed_species = imagenet['category_name'].tolist()
    print(allowed_species)


def get_imagenet():
    if imagenet is None:
        load_imagenet()
    return imagenet


def get_allowed_species():
    if imagenet is None:
        load_imagenet()
    return allowed_species

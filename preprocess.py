
from os.path import join
import argparse

import torch
from tqdm import tqdm

def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))

    formulas_file = join(data_dir, "im2latex_formulas.norm.lst")
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    split_file = join(data_dir, "im2latex_{}_filter.lst".format(split))
    pairs = []
    # transform = transforms.ToTensor()
    with open(split_file, 'r') as f:
        for line in tqdm(f, desc='Generating {} data'.format(split)):
            img_name, formula_id = line.strip('\n').split()
            formula = formulas[int(formula_id)]
            pair = (img_name, formula)
            pairs.append(pair)

    out_file = join(data_dir, "{}.pkl".format(split))
    torch.save(pairs, out_file)
    print("Save {} dataset to {}".format(split, out_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Im2Latex Data Preprocess Program")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    parser.add_argument("-t", "--type_dataset", type=str,
                        default="all", help="Type of dataset to create pkl")
    args = parser.parse_args()

    if args.type_dataset == "all":
        splits = ["validate", "test", "train"]
    else:
        splits = [args.type_dataset]

    for s in splits:
        preprocess(args.data_path, s)

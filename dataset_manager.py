from tetris import Tetris
import argparse
import re
import os
from collections import Counter
import shutil
import numpy as np
from PIL import Image
import random

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--option", type=int, default=0, help="Select mode: 0 (fix given dataset folder), 1 (generate small random dataset from given dataset folder). Default: 0")
parser.add_argument("-l", "--folder_location", type=str, default="samples_r_14 f 20", help="Set the folder location of the dataset that should be fixed/generated from. Default: samples_r_14 f 20")
args = parser.parse_args()

t = Tetris()

arrays = []
scores = []


def _convert_array_to_image(index, score, count):
    # only create folder initially, subsequent calls should add to that folder

    array = np.array(arrays[index])
    image_array = 255 - array * 255
    image = Image.fromarray(image_array)

    file_path = os.path.join(f"{args.folder_location}_fixed", f"sample_{count}_score_{score}.png")
    image.save(file_path)
    # image.show()

def duplicate_entries(score, count, highest_count):
    # obtain all incides with the given score
    indices = [i for i, current_score in enumerate(scores) if current_score == score]
    index = 0

    # going stepwise through arrays[indices[0...n]] duplicating until there are enough entries
    while count < highest_count * 0.9:
        count += 1
        _convert_array_to_image(indices[index], score, count)
        index += 1
        # loop back to start if end is reached
        if index == len(indices):
            index = 0
        

        

if __name__ == "__main__":
    if args.option == 0:
        # fix dataset skewing by duplicating data entries
        i = 0
        for file in os.listdir(args.folder_location):
            # obtain array from file
            array = t._convert_image_to_array(args.folder_location + "\\" + file)
            arrays.append(array)

            # use regex on filename to obtain score
            regex = re.search("score_.*\.png", file)
            regex = regex.group()
            # trim start and end
            score = int(regex[6:-4])
            scores.append(score)
            if i % 1000 == 0:
                print(i, " samples imported")
            i += 1
            # if i >= 100:
            #     break
        
        # dataset imported

        # duplicate to new folder, keep original dataset in tact
        try:
            shutil.rmtree(f"{args.folder_location}_fixed")
        except Exception as e:
            pass
        
        shutil.copytree(args.folder_location, f"{args.folder_location}_fixed")

        # get the largest number of entries of a single score
        counts = Counter(scores)

        highest_count = max(counts.values())

        passed_scores = []

        for score, count in counts.items():
            # prevent duplicate fixes
            if score not in passed_scores:
                # small margin 
                if count < highest_count * 0.9:
                    # duplicate entries
                    duplicate_entries(score, count, highest_count)
                passed_scores.append(score)

    else:
        # obtain a small uniform dataset of 100 entries and generate new folder
        files = [f for f in os.listdir(args.folder_location) if os.path.isfile(os.path.join(args.folder_location, f))]
        random_samples = random.sample(files, min(len(files), 100))

        folder_created = False
        folder_name = f"{args.folder_location}_small"
        i = 1
        while not folder_created:
            try:
                os.mkdir(folder_name)
            except:
                i += 1
                folder_name = f"{args.folder_location}_small_{i}"
            else:
                folder_created = True


        for file in random_samples:
            shutil.copy(os.path.join(args.folder_location, file), folder_name)

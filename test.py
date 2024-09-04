import tetris
import time
import argparse
import os
import re
import numpy as np
import train
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sample_generation", type=int, default=1, help="Determine if either samples should be generated or models should be trained. Default: 1 (= generating sanples)")
parser.add_argument("-r", "--recursion", type=int, default=1, help="Depth of recursion for generating samples. Default: 1")
parser.add_argument("-n", "--n_samples", type=int, default=0, help="Number of samples that should be randomly generated. If 0, then generate all possible samples to set recursion. If > 0, then randomly generate that number of samples up to recursion depth. \nDefault: 0")
args = parser.parse_args()

arrays = []
scores = []

def import_samples(t, folder):
    i = 0
    for file in os.listdir(folder):
        # obtain array from file
        array = t._convert_image_to_array(folder + "\\" + file)
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
        # if i >= 10000:
        #     break


def main():
    t = tetris.Tetris("hi.png")
    if args.sample_generation == 1:
        start_time = time.time()
        
        t.generate_samples(args.recursion, args.n_samples)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Script runtime: {elapsed_time:.2f}.")
    else:
        folder = "samples_r_14"
        import_samples(t, folder)
        reg = train.Train("rf")
        reg.train(arrays, scores)

main()
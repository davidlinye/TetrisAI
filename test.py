import tetris
import time
import argparse
import os
import re
import numpy as np
import train
import joblib
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--option", type=int, default=0, help="Select mode: 0 (running regression training), 1 (generate samples). Default: 0")
parser.add_argument("-r", "--recursion", type=int, default=1, help="Sample generation: Depth of recursion. Default: 1")
parser.add_argument("-n", "--n_samples", type=int, default=0, help="Sample generation: Number of samples that should be randomly generated. If 0, then generate all possible samples to set recursion. If > 0, then randomly generate that number of samples up to recursion depth. \nDefault: 0")
parser.add_argument("-m", "--model", type=str, default="rf", help="Determine the model used for regression task/MCTS. Options: rf (random forest), linear, knn (k-nearest neighbour), cnn (convolutional neural network). Default: rf")
parser.add_argument("-f", "--folder_location", type=str, default="samples_r_14", help="Set the folder location of the samples which are imported for regression task. Default: samples_r_14")
parser.add_argument("-t", "--testset_percentage", type=int, default=0, help="Set the percentage of the test portion of the train/test split for an accuracy plot, or set to 0 to only print accuracy of model. Default: 0 (Give accuracy)")
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
        if i >= 10000:
            break

def main():
    t = tetris.Tetris("hi.png")
    start_time = time.time()
    if args.option == 1:
        
        t.generate_samples(args.recursion, args.n_samples)

    else:
        if args.model not in ["rf", "linear", "knn", "cnn"]:
            print(args.model, " not found, using rf instead")
            args.model = "rf"
        if args.option == 0:
            print("Importing from ", args.folder_location)
            import_samples(t, args.folder_location)
            reg = train.Train(args.model)
            reg.train(arrays, scores, args.testset_percentage)
        else:
            print(f"Option {args.option} does not exist (possible options: 0, 1)")
            return
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Script runtime: {elapsed_time:.2f}.")

if __name__ == "__main__":
    main()
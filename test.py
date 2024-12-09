import tetris
import time
import argparse
import os
import re
import numpy as np
import train
import joblib
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--option", type=int, default=0, help="Select mode: 0 (running regression training), 1 (generate samples), 2 (generate histogram of given dataset). Default: 0")
parser.add_argument("-t", "--target_image", type=str, default="./hi.png", help="Select 2x2 binary image to use as target board for training/testing. Default: ./hi.png")
parser.add_argument("-f", "--forwards", type=int, default=0, help="Sample generation forwards/backwards: 0 (generate from ending position only), 1 (generate forwards from starting and backwards from ending position). Default: 0")
parser.add_argument("-r", "--recursion", type=int, default=1, help="Sample generation: Depth of recursion. Default: 1")
parser.add_argument("-n", "--n_samples", type=int, default=0, help="Sample generation: Number of samples that should be randomly generated. If 0, then generate all possible samples to set recursion. If > 0, then randomly generate that number of samples up to recursion depth. \nDefault: 0")
parser.add_argument("-m", "--model", type=str, default="rf", help="Determine the model used for regression task/MCTS. Options: rf (random forest), linear, knn (k-nearest neighbour), cnn (convolutional neural network). Default: rf")
parser.add_argument("-l", "--folder_location", type=str, default="samples_r_14", help="Set the folder location of the samples which are imported for regression task/histogram generation. Default: samples_r_14")
parser.add_argument("-d", "--dataset_forwards", type=int, default=0, help="Determine if the sample set for regression/histogram generation contains only backwards samples (0) or both forwards and backwards samples (1). Default: 0")
parser.add_argument("-p", "--testset_percentage", type=int, default=0, help="Set the percentage of the test portion of the train/test split for an accuracy plot, or set to 0 to only print accuracy of model. Default: 0 (Give accuracy)")
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
        # if i >= 100:
        #     break

def main():
    t = tetris.Tetris(args.target_image)
    start_time = time.time()
    if args.option == 1:
        
        t.generate_samples(args.recursion, args.n_samples, args.forwards)

    elif args.option == 0:
        if args.model not in ["rf", "linear", "knn", "cnn"]:
            print(args.model, " not found, using rf instead")
            args.model = "rf"
        if args.option == 0:
            print("Importing from ", args.folder_location)
            import_samples(t, args.folder_location)
            reg = train.Train(args.model)
            reg.train(arrays, scores, args.folder_location, args.testset_percentage, args.dataset_forwards)
        else:
            print(f"Option {args.option} does not exist (possible options: 0, 1)")
            return
    
    else:
        print("Importing from ", args.folder_location)
        import_samples(t, args.folder_location)
        if args.dataset_forwards == 0:
            plt.hist(scores, bins=range(1,22), edgecolor='black')

            # Add titles and labels
            plt.title(f'Score distribution {args.folder_location}')
            plt.xlabel('Score Ranges')
            plt.ylabel('Frequency')
            plt.xticks(range(1, max(scores) + 1), rotation=-45)
            plt.savefig(f"graphs/histogram_{args.folder_location}.png")
        else:
            scores_begin = [score for score in scores if score < 50]
            scores_end = [score for score in scores if score > 50]
            plt.hist(scores_begin, bins=range(1,max(scores_begin) + 2), edgecolor='black')

            # Add titles and labels
            plt.title(f'Score distribution {args.folder_location} backwards')
            plt.xlabel('Score Ranges')
            plt.ylabel('Frequency')
            plt.xticks(range(1, max(scores_begin) + 1), rotation=-45)
            plt.savefig(f"graphs/histogram_{args.folder_location}_begin.png")

            plt.clf()

            plt.hist(scores_end, bins=range(min(scores_end), 101), edgecolor='black')

            # Add titles and labels
            plt.title(f'Score distribution {args.folder_location} forwards')
            plt.xlabel('Score Ranges')
            plt.ylabel('Frequency')
            plt.xticks(range(min(scores_end), 101), rotation=-45)
            plt.savefig(f"graphs/histogram_{args.folder_location}_end.png")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Script runtime: {elapsed_time:.2f}.")

if __name__ == "__main__":
    main()
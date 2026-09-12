import pandas as pd
from matrix_lib import MatrixFloat, e_types
import argparse
import math
import random
pd.set_option("future.no_silent_downcasting", True)
def relu(data):
    for i in range(len(data)):
        if (data[i] < 0):
            data[i] = 0
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "--layers",
                type = int,
                nargs="+",
                default= [5, 5],
                help="List of layer sizes (ex: 5 5, is 2 layers of 5 neurons)"
                )
        parser.add_argument(
                "--epochs",
                type = int,
                default= 100,
                help="How much epochs you want the learning to be"
                )
        parser.add_argument(
                "--learning-rate",
                type = int,
                default= 0.1,
                help="How fast you want it to learn, the more its fast, the less its precise"
                )
        args = parser.parse_args()
        epochs = args.epochs
        layers = args.layers
        learning_rate = args.learning_rate
        #as data.csv doesnt contain header we cant read it
        # but we know 1 will be M or B
        df = pd.read_csv('data.csv', header = None)
        df = df.drop(columns = [0])
        df = df.dropna()
        df = df[(df[1] == "M") | (df[1] == "B")]
        df[1] = df[1].replace('M', 1)
        df[1] = df[1].replace('B', 0)
        guess = df[1]
        dataset = df.drop(columns = [1])
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min()) 
        #80% of the dataset for learning
        split = int(len(dataset) * 0.8)
        learn_df = dataset.iloc[:split]
        # 100 - 80 (20) for exam
        exam_df = dataset.iloc[split:]
        input_size = len(dataset.columns)
        #litteraly makes a Matrix for every line of the dataset
        input_matrix_list = [MatrixFloat(input_size, 1, e_types.NO_TYPE, [value for column, value, in learn_df.iloc[i].items()], -1, -1) for i in range(len(learn_df))]
        layers_bias_matrices = [MatrixFloat(layer, 1, e_types.NO_TYPE, [0.0 for i in range(layer)], -1, -1) for layer in layers]
        layers_weight_matrices = [MatrixFloat(1, 5, e_types.NO_TYPE, [random.gauss(0.0, math.sqrt(2.0 / input_size)) for _ in range(layer * input_size)], -1, -1)
                                  for layer in layers]
        for epoch in range(epochs):
            for i in range(len(input_matrix_list)):
                X = input_matrix_list[i]
                Y = guess[i]
                for layer in layers:
                    Z = layers_weight_matrices[layer].multiply_mat(X)
                    Z = Z.add_mat(layers_bias_matrices[layer])
                    data = Z.getData()
                    relu(data)
                    X = data
                Y = 1 / (1 + math.exp(-X))
                print(Y)


    except Exception as e:
        print (e)



if __name__ == "__main__":
    main()

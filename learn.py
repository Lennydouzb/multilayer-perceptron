import pandas as pd
from matrix_lib import MatrixFloat, e_types
import argparse
import math
import random
import matplotlib.pyplot as plt
pd.set_option("future.no_silent_downcasting", True)
def relu(data):
    tmp = []
    for i in data:
        if i < 0:
            tmp.append(0)
        else:
            tmp.append(i)
    return tmp
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "--layers",
                type = int,
                nargs="+",
                default= [5, 5],
                help="List of layer sizes (ex: 5 5 6, is 2 layers of 5 neurons and one of 6, then output)"
                )
        parser.add_argument(
                "--epochs",
                type = int,
                default= 100,
                help="How much epochs you want the learning to be"
                )
        parser.add_argument(
                "--learning-rate",
                type = float,
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
        #we add the output layer (1 neuron)
        architecture = layers + [1]
        #litteraly makes a Matrix for every line of the dataset
        input_matrix_list = [MatrixFloat(input_size, 1, e_types.NO_TYPE, [value for column, value, in learn_df.iloc[i].items()], -1, -1) for i in range(len(learn_df))]
        input_matrix_list_exam = [MatrixFloat(input_size, 1, e_types.NO_TYPE, [value for column, value, in exam_df.iloc[i].items()], -1, -1) for i in range(len(exam_df))]

        layers_weight_matrices = []
        layers_bias_matrices = []
        current_input_size = input_size
        for layer in architecture:
            bias = MatrixFloat(layer, 1, e_types.NO_TYPE, [0.0] * layer, -1, -1)
            layers_bias_matrices.append(bias)
            data = [random.gauss(0.0, math.sqrt(2.0 / current_input_size)) for _ in range(current_input_size * layer)]
            weight = MatrixFloat(layer, current_input_size, e_types.NO_TYPE, data, -1, -1)
            layers_weight_matrices.append(weight)
            current_input_size = layer
        epochs_plot = []
        error_plot = []
        error_plot_exam = []
        for epoch in range(epochs):
            epochs_plot.append(epoch)
            error_sum = 0
            #
            #   Learning Phase !!!!
            #
            for i in range(len(input_matrix_list)):
                X = input_matrix_list[i]
                Y = guess.iloc[i]
                #memory matrices
                A_matrices = [X]
                Z_matrices = []
                for j in range(len(architecture)):
                    Z = layers_weight_matrices[j].multiply_mat(A_matrices[-1])
                    Z = Z.add_mat(layers_bias_matrices[j])
                    Z_matrices.append(Z)
                    data = Z.getData()
                    if j == len(architecture) - 1:
                        prediction = 1 / (1 + math.exp(-data[0]))
                        error_sum += abs(prediction - Y)
                        tmp = MatrixFloat(1, 1, e_types.NO_TYPE, [prediction], -1, -1)
                    else:
                        relu_data = relu(data)
                        tmp = MatrixFloat(len(relu_data), 1, e_types.NO_TYPE, relu_data, -1, -1)
                    A_matrices.append(tmp)

                error = A_matrices[-1].sub_scalar(Y)
                for j in range(len(architecture) - 1, -1, -1):
                    transposed_A = A_matrices[j].transpose()
                    gradient = error.multiply_mat(transposed_A)

                    if j > 0:
                        #passing the error for the next layer
                        transposed_weight = layers_weight_matrices[j].transpose()
                        propagated_error = transposed_weight.multiply_mat(error)
                        error_matrix = MatrixFloat(len(Z_matrices[j - 1].getData()), 1, e_types.NO_TYPE, [1.0 if val > 0 else 0.0 for val in Z_matrices[j - 1].getData()], -1, -1)
                        # this is hadamard, if they were involved (relu wisely) they have error, if not, they do not have error
                        next_error = propagated_error.hadamard(error_matrix)
                    layers_weight_matrices[j] = layers_weight_matrices[j].sub_mat(gradient.multiply_scalar(learning_rate))
                    layers_bias_matrices[j] = layers_bias_matrices[j].sub_mat(error.multiply_scalar(learning_rate))
                    if j > 0:
                        error = next_error
            #
            #   Exam phase !!!
            #
            error_sum_exam = 0
            for i in range(len(input_matrix_list_exam)):
                A_matrices_exam = [input_matrix_list_exam[i]]
                Y = guess.iloc[split + i] 
                for j in range(len(architecture)):
                    Z = layers_weight_matrices[j].multiply_mat(A_matrices_exam[-1])
                    Z = Z.add_mat(layers_bias_matrices[j])
                    data = Z.getData()
                    if j == len(architecture) - 1:
                        prediction = 1 / (1 + math.exp(-data[0]))
                        error_sum_exam += abs(prediction - Y)
                        tmp = MatrixFloat(1, 1, e_types.NO_TYPE, [prediction], -1, -1)
                    else:
                        relu_data = relu(data)
                        tmp = MatrixFloat(len(relu_data), 1, e_types.NO_TYPE, relu_data, -1, -1)
                    A_matrices_exam.append(tmp)
            error_plot.append(error_sum / len(input_matrix_list))
            error_plot_exam.append(error_sum_exam / len(input_matrix_list_exam))
        plt.plot(epochs_plot, error_plot, color='red', label='train loss')
        plt.plot(epochs_plot, error_plot_exam, color='yellow', label='exam loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()   
        plt.show()

    except Exception as e:
        print (e)



if __name__ == "__main__":
    main()

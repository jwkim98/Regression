import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import math
import random

num_data = 100000

# Generating data
# Split the data into a training set of first 8000 instances, validation set of next 1000 instances
# and a test set of the rest 1000 instances
# uniformly distributed random noise (-3, 3) is used to generate x1_array and x2_array
x1_array = np.random.uniform(-3, 3, num_data)
x2_array = np.random.uniform(-3, 3, num_data)
input_array = np.empty([num_data])

for i in range(0,num_data):
    input = 2.0 + 1.0*x1_array[i] + 0.5*pow(x1_array[i],2) + 0.25*pow(x1_array[i],3) + 0.5*pow(x2_array[i],2) + np.random.normal(0,1)
    input_array[i] = input

train_set_y = input_array[:8000]
validation_set_y = input_array[8000:9000]
test_set_y = input_array[9000:]

train_set_x1 = x1_array[:8000]
validation_set_x1 = x1_array[8000:9000]
test_set_x1 = x1_array[9000:]

train_set_x2 = x2_array[:8000]
validation_set_x2 = x2_array[8000:9000]
test_set_x2 = x2_array[9000:]


# 2. We learn (3 + 5)!/(3!*5!) parameters here which is 56. Therefore, we learn 56 parameters
poly = PolynomialFeatures(5)


# 3. We want to regularize y = a + b*x1 + c*x1^2 + d*x1^3 + e*x2^2
# Using polynomial regression, we should minimize (1/2)(X*theta - y)^T(X*theta - y)
# Which gives us derivative X^T(X*theta - y)
def polynomial(params, x1, x2):
    val = params[0] + params[1]*x1 + params[2]*pow(x1,2) + params[3]*pow(x1,3) + params[4]*pow(x2,2)
    #print("val: " + str(val))
    return val


# partial derivative for all parameters
def polynomial_derivative(params, x1, x2, index):
    val = 0
    if index == 0:
        val =  1
    elif index == 1:
        val = x1
    elif index == 2:
        val = pow(x1,2)
    elif index == 3:
        val = pow(x1,3)
    elif index == 4:
        val = pow(x2,2)
    return val

def evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y):
    sum = 0
    for i in range(0, len(test_set_x1)):
        sum += pow(polynomial(params,test_set_x1[i], test_set_x2[i]) - test_set_y[i], 2)
    # print(sum)
    return math.sqrt(sum/len(test_set_x1))


def linear_regression(epochs, learning_rate):
    num_params = 5
    params = [0,0,0,0,0]
    validation_error_list = []
    train_error_list = []


    for j in range(0, epochs):
        for k in range(0, num_params):
            update = (polynomial(params, train_set_x1[j], train_set_x2[j]) - train_set_y[j]) * polynomial_derivative(params, train_set_x1[j], train_set_x2[j], k)
            params[k] = params[k] - learning_rate*update
        validation_error = evaluate_rmse(params, validation_set_x1, validation_set_x2, validation_set_y)
        train_error = evaluate_rmse(params, train_set_x1, train_set_x2, train_set_y)
        validation_error_list.append(validation_error)
        train_error_list.append(train_error)

        # print("Epoch: " + str(j) + " validation_error : " + str(validation_error) + " train_error : " + str(train_error))

    test_rmse = evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y)
    print("Test rmse gradient_descent: " + str(test_rmse))

    plt.plot(list(range(0, epochs)), train_error_list, c = 'b')
    plt.plot(list(range(0, epochs)), validation_error_list, c = 'r')
    plt.savefig("Gradient_descent.png")
    plt.show()



def ridge_regression(epochs, learning_rate, alpha):
    num_params = 5
    params = [0,0,0,0,0]
    validation_error_list = []
    train_error_list = []

    for j in range(0, epochs):
        for k in range(0, num_params):
            # update = X^t(X*theta - y) + alpha * theta
            theta = 0 if k == 0 else params[k]
            update = (polynomial(params, train_set_x1[j], train_set_x2[j]) - train_set_y[j]) * polynomial_derivative(params, train_set_x1[j], train_set_x2[j], k) + alpha*theta
            params[k] -= learning_rate*update

        validation_error = evaluate_rmse(params, validation_set_x1, validation_set_x2, validation_set_y)
        train_error = evaluate_rmse(params, train_set_x1, train_set_x2, train_set_y)
        validation_error_list.append(validation_error)
        train_error_list.append(train_error)

        # print("Epoch: " + str(i) + "validation_error : " + str(validation_error) + "train_error : " + str(train_error))

    test_rmse = evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y)
    print("Test rmse ridge_regression: " + str(test_rmse))

    plt.plot(list(range(0, epochs)), train_error_list, c = 'b')
    plt.plot(list(range(0, epochs)), validation_error_list, c = 'r')
    plt.savefig("ridge_regression.png")
    plt.show()

def lasso_regression(epochs, learning_rate, alpha):
    num_params = 5
    params = [0,0,0,0,0]
    validation_error_list = []
    train_error_list = []

    for j in range(0, epochs):
        for k in range(0, num_params):

            theta = 0
            if k > 0:
                if params[k] > 0:
                    theta = 1
                elif params[k] < 0:
                    theta = -1
                else:
                    theta = 0

            update = (polynomial(params, train_set_x1[j], train_set_x2[j]) - train_set_y[j]) * polynomial_derivative(params, train_set_x1[j], train_set_x2[j], k) + alpha*theta
            params[k] -= learning_rate*update

        validation_error = evaluate_rmse(params, validation_set_x1, validation_set_x2, validation_set_y)
        train_error = evaluate_rmse(params, train_set_x1, train_set_x2, train_set_y)
        validation_error_list.append(validation_error)
        train_error_list.append(train_error)

        # print("Epoch: " + str(j) + " validation_error : " + str(validation_error) + "train_error : " + str(train_error))

    test_rmse = evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y)
    print("Test rmse lasso_regression: " + str(test_rmse))

    plt.plot(list(range(0, epochs)), train_error_list, c = 'b')
    plt.plot(list(range(0, epochs)), validation_error_list, c = 'r')
    plt.savefig("lasso_regression.png")
    plt.show()


def linear_regression_batch(epochs, learning_rate, batch_size):
    num_params = 5
    params = [0,0,0,0,0]
    validation_error_list = []
    train_error_list = []

    tuple_list = []

    for i in range(0, len(train_set_x1)):
        tuple_list.append((train_set_x1[i], train_set_x2[i], train_set_y[i]))

    random.shuffle(tuple_list)
    train_data_list = tuple_list[:batch_size]

    for i in range(0, epochs):
        for j in range(0, len(train_data_list)):
            for k in range(0, num_params):
                update = (polynomial(params, train_data_list[j][0], train_data_list[j][1]) - train_data_list[j][2]) * polynomial_derivative(params,train_data_list[j][0], train_data_list[j][1], k)
                params[k] -= learning_rate*update

        validation_error = evaluate_rmse(params, validation_set_x1, validation_set_x2, validation_set_y)
        train_error = evaluate_rmse(params, train_set_x1, train_set_x2, train_set_y)
        validation_error_list.append(validation_error)
        train_error_list.append(train_error)

        # print("Epoch: " + str(i) + "validation_error : " + str(validation_error) + "train_error : " + str(train_error))
    
    test_rmse = evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y)
    print("Test rmse minibatch_gradient: " + str(test_rmse))

    plt.plot(list(range(0, epochs)), train_error_list, c = 'b')
    plt.plot(list(range(0, epochs)), validation_error_list, c = 'r')
    plt.savefig("minibatch_gradient_descent.png")
    plt.show()

def ridge_regression_batch(epochs, learning_rate, alpha, batch_size):
    num_params = 5
    params = [0,0,0,0,0]
    validation_error_list = []
    train_error_list = []

    tuple_list = []

    for i in range(0, len(train_set_x1)):
        tuple_list.append((train_set_x1[i], train_set_x2[i], train_set_y[i]))

    random.shuffle(tuple_list)
    train_data_list = tuple_list[:batch_size]

    for i in range(0, epochs):
        for j in range(0, len(train_data_list)):
            for k in range(0, num_params):
                # update = X^t(X*theta - y) + alpha * theta
                theta = 0 if k == 0 else params[k]
                update = (polynomial(params, train_data_list[j][0], train_data_list[j][1]) - train_data_list[j][2]) * polynomial_derivative(params,train_data_list[j][0], train_data_list[j][1], k) + alpha*theta
                params[k] -= learning_rate*update

        validation_error = evaluate_rmse(params, validation_set_x1, validation_set_x2, validation_set_y)
        train_error = evaluate_rmse(params, train_set_x1, train_set_x2, train_set_y)
        validation_error_list.append(validation_error)
        train_error_list.append(train_error)

        # print("Epoch: " + str(i) + "validation_error : " + str(validation_error) + "train_error : " + str(train_error))
    
    test_rmse = evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y)
    print("Test rmse ridge_regression: " + str(test_rmse))

    plt.plot(list(range(0, epochs)), train_error_list, c = 'b')
    plt.plot(list(range(0, epochs)), validation_error_list, c = 'r')
    plt.savefig("minibatch_ridge_regression.png")
    plt.show()


def lasso_regression_batch(epochs, learning_rate, alpha, batch_size):
    num_params = 5
    params = [0,0,0,0,0]
    validation_error_list = []
    train_error_list = []

    tuple_list = []

    for i in range(0, len(train_set_x1)):
        tuple_list.append((train_set_x1[i], train_set_x2[i], train_set_y[i]))

    random.shuffle(tuple_list)
    train_data_list = tuple_list[:batch_size]

    for i in range(0, epochs):
        for j in range(0, len(train_data_list)):
            for k in range(0, num_params):

                theta = 0
                if k > 0:
                    if params[k] > 0:
                        theta = 1
                    elif params[k] < 0:
                        theta = -1
                    else:
                        theta = 0

                theta = 0 if k == 0 else params[k]
                update = (polynomial(params, train_data_list[j][0], train_data_list[j][1]) - train_data_list[j][2]) * polynomial_derivative(params,train_data_list[j][0], train_data_list[j][1], k) + alpha*theta
                params[k] -= learning_rate*update

        validation_error = evaluate_rmse(params, validation_set_x1, validation_set_x2, validation_set_y)
        train_error = evaluate_rmse(params, train_set_x1, train_set_x2, train_set_y)
        validation_error_list.append(validation_error)
        train_error_list.append(train_error)

        # print("Epoch: " + str(i) + "validation_error : " + str(validation_error) + "train_error : " + str(train_error))
    
    test_rmse = evaluate_rmse(params, test_set_x1, test_set_x2, test_set_y)
    print("Test rmse lasso_regression_minibatch : " + str(test_rmse))

    plt.plot(list(range(0, epochs)), train_error_list, c = 'b')
    plt.plot(list(range(0, epochs)), validation_error_list, c = 'r')
    plt.savefig("minibatch_lasso_regression.png")
    plt.show()

    
batch_size = 100

linear_regression(70, 0.001)    
ridge_regression(70, 0.001, 0.01)
lasso_regression(70, 0.001, 0.01)
linear_regression_batch(70, 0.001, batch_size)
ridge_regression_batch(70, 0.001, 0.01, batch_size)
lasso_regression_batch(70, 0.001, 0.01, batch_size)
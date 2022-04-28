import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)

def plot_decision_boundary(pred_func, X, y, color_ind, alph, linewd):
    # Set min and max values and give it some padding
    padd = 5
    x_min, x_max = X[:, 0].min() - padd,  padd
    y_min, y_max = X[:, 1].min() - padd,  padd
    h =0.04
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contour(xx, yy, Z, alpha = alph,  levels=14, linewidths=linewd, colors=color_ind)


def Compute_decision_boundary(pred_func, X, y, color_ind):

    # Set min and max values and give it some padding
    padd = 5
    x_min, x_max = X[:, 0].min() - padd,  padd
    y_min, y_max = X[:, 1].min() - padd,  padd
    h = 0.04
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    return Z

# data initialization
X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y = np.array([[0,1,1,0]]).T

class BinNN(object):
    def __init__(self, n_x, n_h, n_y, ww):
        """
        n_x: int
            number of neurons in input layer
        n_h: int
            number of neurons in hidden layer
        n_y: int
            number of neurons in output layer
        """

        bin_w = np.array(ww)
        a1 = n_x * n_h
        a2 = a1 + n_h
        a3 = a2 + n_h * n_y
        self.W1 = bin_w[:a1].reshape(n_x, n_h)
        self.b1 = bin_w[a1:a2]
        self.W2 = bin_w[a2:a3].reshape(n_h, n_y)

    def _sigmoid(self, Z):
        # sigmoid activation function
        return (1 / (1 + np.exp(-Z)))

    def feedforward(self, X):
        """performing the feedforward pass
        """

        # first hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._sigmoid(self.Z1)
        # second layer
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = self._sigmoid(self.Z2)

        return self.A2

    def predict(self, X):
        A2 = self.feedforward(X)
        predictions = np.round(A2)
        return predictions


inputX = 2
hiddenD = 3
outputD = 1
totalD = inputX * hiddenD + hiddenD + hiddenD*outputD
print("All the binary weights: {} for training".format(totalD))

# ------------Unique decision boundaries for sign pruned network (no half-half constrain)------------
prunerateall = [0]
for index in range(len(prunerateall)):
    prunerate = prunerateall[index]
    sum_rem = int(totalD - prunerate)
    from itertools import product
    l = [-1, 0, 1]
    combin = list(product(l, repeat=totalD))
    combin_arr = np.array(combin)
    sum_all = np.abs(combin_arr).sum(1)
    indx = np.where(sum_all==sum_rem)  # pick up all possible masked net
    combin = combin_arr[indx[0]]
    combin_len = combin.shape[0]
    print("All combinations: {} for training".format(combin_len))
    color = plt.cm.rainbow(np.linspace(0, 1, combin_len))
    first = True
    for ind, bin_w in enumerate(combin):
        Binary_model = BinNN(n_x=inputX, n_h=hiddenD, n_y=outputD, ww = bin_w)  # without training
        color_ind = color[ind]
        Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y, color_ind.reshape(-1,4))

        if Z.sum() != 0 and Z.sum() != Z.shape[0]:
            if first:
                all_decisionB = Z
                All_bin_w = bin_w
            else:
                all_decisionB = np.concatenate((all_decisionB, Z), 1)
                All_bin_w = np.vstack((All_bin_w, bin_w))

            first = False

    all_decisionB = np.array(all_decisionB)
    signall_decisionB = np.sign((all_decisionB - 0.5))
    q = signall_decisionB.shape[0]
    distH = 0.5 * (q - np.dot(signall_decisionB.transpose(), signall_decisionB ))
    distH[distH == q] = 0
    distH_used = distH
    aa = distH == 0
    U = np.triu(aa,1)
    U_sum = U.sum(0)
    Unique_decision = len(U_sum) - (U_sum != 0).sum()
    print("Unique decision boundaries: {} for training".format(Unique_decision))

    index = np.where(U_sum == 0)[0]
    hist = np.zeros([1, len(index)])
    i = 0
    first = True
    for ii in index:
        aa = All_bin_w[distH_used[ii] == 0]   # pick up all weight combinations
        hist[0, i] =(distH_used[ii] == 0).sum()
        i+=1

        if first:
            Unique_weight = aa[0]
        else:
            Unique_weight = np.vstack((Unique_weight, aa[0]))
        first = False
    # save all weights with unique decision boundaries, for later plotting
    save_file_path1 = './Sign_Unique_weight.npy'
    np.save(save_file_path1, Unique_weight)

# ------------Unique decision boundaries for half-half pruned network------------
for index in range(len(prunerateall)):
    prunerate =  prunerateall[index]
    print("Half_half with prune rate: {} for training".format(prunerate))
    sum_rem = int(totalD - prunerate)
    from itertools import product
    l = [-1, 0, 1]
    combin = list(product(l, repeat = totalD))
    combin_arr = np.array(combin)
    sum_all = np.abs(combin_arr).sum(1)
    indx = np.where(sum_all==sum_rem)  # pick up all possible masked net
    combin = combin_arr[indx[0]]
    combin_arr = combin
    sum_all = (combin_arr).sum(1)
    sum_all_abs = np.abs(sum_all)
    min_sum = sum_all_abs.min()
    indx = np.where(sum_all_abs==min_sum)  # pick up all possible masked net
    combin = combin_arr[indx[0]]
    combin_len = combin.shape[0]
    print("Half_half All combinations: {} for training".format(combin_len))
    color = plt.cm.rainbow(np.linspace(0, 1, combin_len))
    first = True
    first0 = True
    for ind, bin_w in enumerate(combin):
        Binary_model = BinNN(n_x=inputX, n_h=hiddenD, n_y=outputD, ww = bin_w)  # without training
        color_ind = color[ind]
        Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y, color_ind.reshape(-1,4))

        if Z.sum() != 0 and Z.sum() != Z.shape[0]:
            if first:
                all_decisionB = Z
                All_bin_w = bin_w
            else:
                all_decisionB = np.concatenate((all_decisionB, Z), 1)
                All_bin_w = np.vstack((All_bin_w, bin_w))

            first = False
    all_decisionB = np.array(all_decisionB)
    signall_decisionB = np.sign((all_decisionB - 0.5))
    q = signall_decisionB.shape[0]
    distH = 0.5 * (q - np.dot(signall_decisionB.transpose(), signall_decisionB ))
    distH[distH == q] = 0
    distH_used = distH
    aa = distH == 0
    U = np.triu(aa,1)
    U_sum = U.sum(0)
    Unique_decision = len(U_sum) - (U_sum != 0).sum()
    print("Half_half Unique decision boundaries: {} for training".format(Unique_decision))

    index = np.where(U_sum == 0)[0]
    hist = np.zeros([1, len(index)])
    i = 0
    first = True
    for ii in index:
        aa = All_bin_w[distH_used[ii] == 0]   # pick up all weight combinations
        hist[0, i] =(distH_used[ii] == 0).sum()
        i+=1

        if first:
            Unique_weight_half = aa[0]
        else:
            Unique_weight_half = np.vstack((Unique_weight_half, aa[0]))

        first = False
    # save all weights with unique decision boundaries, for later plotting
    save_file_path2 = './Sign_Half_half_Unique_weight.npy'
    np.save(save_file_path2, Unique_weight_half)


'''
Following is: plotting the unique decision boundaries
'''
alph = 1
color = 'b'
ax = plt.figure(figsize=(5.0, 4.0) )

# ------------sign pruned network decision boundaries------------
first = True
for ind, bin_w in enumerate(Unique_weight):
    print("current training ind: {} for training".format(ind))

    Binary_model = BinNN(n_x=inputX, n_h=hiddenD, n_y=outputD, ww = bin_w, learning_rate=0.01)  # without training

    Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y, color)

    if Z.sum() != 0 and Z.sum() != Z.shape[0]:
        if first:
            all_decisionB = Z
            All_bin_w = bin_w
        else:
            all_decisionB = np.concatenate((all_decisionB, Z), 1)
            All_bin_w = np.vstack((All_bin_w, bin_w))

        first = False

all_decisionB = np.array(all_decisionB)
signall_decisionB_without = np.sign((all_decisionB - 0.5))

# ------------half pruned network decision boundaries------------
first = True
for ind, bin_w in enumerate(Unique_weight_half):
    print("current training ind: {} for training".format(ind))

    Binary_model = BinNN(n_x=inputX, n_h=hiddenD, n_y=outputD, ww = bin_w, learning_rate=0.01)  # without training

    Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y, color)

    if Z.sum() != 0 and Z.sum() != Z.shape[0]:
        if first:
            all_decisionB0 = Z
            All_bin_w_half = bin_w
        else:
            all_decisionB0 = np.concatenate((all_decisionB0, Z), 1)
            All_bin_w_half = np.vstack((All_bin_w_half, bin_w))

        first = False

all_decisionB0 = np.array(all_decisionB0)
signall_decisionB_half = np.sign((all_decisionB0 - 0.5))

# intersect: the difference set of unique decision boundaries between sign and half-half
distH = 0.5 * (q - np.dot(signall_decisionB_without.transpose(),  signall_decisionB_half))
distH[distH == q] = 0
distH_used = distH
aa = distH == 0
bb= aa.sum(1)
index = np.where(bb == 0)[0]
first = True
for ii in index:
    aa = All_bin_w[ii]   # pick up all weight combinations

    if first:
        Unique_black = aa
    else:
        Unique_black = np.vstack((Unique_black, aa))

    first = False

# plot the unique decision boundaries by half-half
color = 'r'
linewd = 0.5
for ind, bin_w in enumerate(Unique_weight_half):
    Binary_model = BinNN(n_x=inputX, n_h=hiddenD, n_y=outputD, ww = bin_w, learning_rate=0.01)  # without training
    plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)

# plot the difference set of unique decision boundaries between sign and half-half
linewd = 0.5
alph = 0.3
color = 'k'
for ind, bin_w in enumerate(Unique_black):
    Binary_model = BinNN(n_x=inputX, n_h=hiddenD, n_y=outputD, ww = bin_w, learning_rate=0.01)  # without training
    plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)

# plot setting
x_ticks =[-5, 0, 5]
x_ticks_lable = [ '-5',  '0',  '5']
plt.xticks(x_ticks, x_ticks_lable, fontsize=12)

y_ticks =[-5, 0, 5]
y_ticks_lable = [ '-5',  '0',  '5']
plt.yticks(y_ticks, y_ticks_lable, fontsize=12)

plt.xlabel(r'$\mathtt{x}_1$ input value',font1)
plt.ylabel('x$_2$ input value',font1)
plt.tick_params(labelsize=13.5)
plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt

lam = 0.06
def find_y_new (X_s, Y_s, w, b):
    # w is a scale vector, b is an offset vector
    return np.multiply(Y_s, np.multiply(w, X_s)) + np.multiply(b, X_s)

def create_data ():
    # Create X_s (training set)
    a = np.arange(-3, -0.8, 0.2)
    b = np.arange(-0.5, 0.5, 0.5) 
    c = np.arange(3, 5.2, 0.2)
    X_s = np.concatenate((a, b, c))

    # Create Y_s (training set)
    Y_s = np.sin(X_s)

    noise_s = np.random.normal(0, 0.1, len(Y_s))
    Y_s += noise_s

    # Create X_t (test set)
    X_t = np.arange(-5, 5, 0.35)

    # Create Y_t (test set)
    Y_t = np.sin(X_t) + 1

    noise_t = np.random.normal(0, 0.1, len(Y_t))
    Y_t += noise_t

    # Uncomment below if wish to see the graph
    #plt.plot(X_s, Y_s, 'r*', X_t, Y_t, 'bo-')
    #plt.show()
    return X_s, Y_s, X_t, Y_t

def check_converge (w, b, w_old, b_old):
    for i in range(len(w)):
        if abs(w[i] - w_old[i]) > 0.000001:
            return False

    for i in range(len(b)):
        if abs(b[i] - b_old[i]) > 0.000001:
            return False

    # w and b converge when the code reaches here
    return True

def find_sum (X):
    ret = 0
    for i in range(len(X)):
        ret += X[i]
    return ret

def find_difference (Y_s, Y_t):
    Y_s_max = Y_s[0]
    Y_s_min = Y_s[0]
    for i in range(len(Y_s)):
        if Y_s[i] > Y_s_max:
            Y_s_max = Y_s[i]
        elif Y_s[i] < Y_s_min:
            Y_s_min = Y_s[i]
    
    Y_t_max = Y_t[0]
    Y_t_min = Y_t[0]
    for i in range(len(Y_t)):
        if Y_t[i] > Y_t_max:
            Y_t_max = Y_t[i]
        elif Y_t[i] < Y_t_min:
            Y_t_min = Y_t[i]

    return Y_t_max - Y_s_min, Y_t_min - Y_s_max

# Find linear kernel between matrix x and y
def find_kernel (x, y):
    ret = np.zeros(shape=(len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            ret[i][j] = x[i] * y[j]

    return ret


# Do transfer learning by conditional distribution matching
def cdm (X_s, Y_s, X_t, Y_t):
    w = np.ones(len(X_s))
    b = np.zeros(len(X_s))

    w_old = np.zeros(len(X_s))
    b_old = np.ones(len(X_s))

    Y_new = find_y_new(X_s, Y_s, w, b)

    K_xx = find_kernel(X_s, X_s)
    K_ynew = find_kernel(Y_new, Y_new)
    K_ynewyt = find_kernel(Y_new, Y_t)
    K_xtx = find_kernel(X_t, X_s)
    K_tt = find_kernel(X_t, X_t)


    L_s = find_kernel(X_s, X_s)
    R = L_s * np.linalg.inv(L_s + lam * np.eye(len(L_s)))

    # Repeat this loop until w, b converge
    num_itr = 0
    MAX_ITR = 100
    step_size = 0.00001
    while num_itr < MAX_ITR:
        num_itr += 1
        for i in range(len(w)):
            w_old[i] = w[i]
            b_old[i] = b[i];

        sigma = np.std(Y_s)

        # Construct D_p, E_p
        D_p = np.zeros(shape=(len(Y_new), len(Y_new)))
        E_p = np.zeros(shape=(len(Y_new), len(Y_t)))
        for p in range(len(w)):
            for i in range(len(Y_new)):
                for j in range(len(Y_new)):
                    D_p[i][j] = - (1 / (sigma ** 2)) * (Y_new[i] - Y_new[j]) * (Y_s[i] * R[i][p] - Y_s[j] * R[j][p])

            for i in range(len(Y_new)):
                for j in range(len(Y_t)):
                    E_p[i][j] = - (1 / (sigma ** 2)) * (Y_new[i] - Y_t[j]) * Y_s[i] * R[i][p]

        # Update values in w
        for p in range(len(w)):
            dL_over_dK = np.transpose(np.matmul(np.matmul(np.linalg.inv(K_xx + lam * np.eye(len(K_xx))), np.transpose(K_xx)), np.linalg.inv(K_xx + lam *np.eye(len(K_xx)))))
            K_mul_Dp = np.multiply(K_ynew, D_p) #Used to be D_p[p]
            dL_over_dKtilda = np.transpose(2 * np.matmul(np.matmul(np.linalg.inv(K_xx + lam * np.eye(len(K_xx))), np.transpose(K_xtx)), np.linalg.inv(K_tt + lam * np.eye(len(K_tt)))))
            Ktilda_mul_Ep = np.multiply(K_ynewyt, E_p[p])
            gradient = np.trace(np.matmul(dL_over_dK, K_mul_Dp)) - np.trace(np.matmul(dL_over_dKtilda, Ktilda_mul_Ep))
            w[p] += step_size * gradient
        
       
        # Construct D_ptilda and E_ptilda
        D_ptilda = np.zeros(shape=(len(Y_new), len(Y_new)))
        E_ptilda = np.zeros(shape=(len(Y_new), len(Y_t)))
        for p in range(len(b)):
            for i in range(len(Y_new)):
                for j in range(len(Y_new)):
                    D_ptilda[i][j] = - (1 / (sigma ** 2)) * (Y_new[i] - Y_new[j]) * (R[i][p] - R[j][p])

            for i in range(len(Y_new)):
                for j in range(len(Y_t)):
                    E_ptilda[i][j] = - (1 / (sigma ** 2)) * (Y_new[i] - Y_t[j]) * R[i][p]

        # Update values in b
        for p in range(len(b)):
            dL_over_dK = np.transpose(np.matmul(np.matmul(np.linalg.inv(K_xx + lam * np.eye(len(K_xx))), np.transpose(K_xx)), np.linalg.inv(K_xx + lam * np.eye(len(K_xx)))))
            K_mul_D_ptilda = np.multiply(K_ynew, D_ptilda)
            dL_over_dKtilda = np.transpose(2 * np.matmul(np.matmul(np.linalg.inv(K_xx + lam * np.eye(len(K_xx))), np.transpose(K_xtx)), np.linalg.inv(K_tt + lam * np.eye(len(K_tt)))))
            Ktilda_mul_E_ptilda = np.multiply(K_ynewyt, E_ptilda[p])
            gradient = np.trace(np.matmul(dL_over_dK, K_mul_D_ptilda)) - np.trace(np.matmul(dL_over_dKtilda, Ktilda_mul_E_ptilda))
            b[p] += step_size * gradient
            

        # Bound w and b i.e. if they are not within the range of 0-1 or 0-2
        # give them the corresponding values at the limit
        """
        for i in range(len(w)):
            if w[i] < 0:
                w[i] = 0
            elif w[i] > 1:
                w[i] = 1

        b_max, b_min = find_difference(Y_s, Y_t)
        for i in range(len(b)):
            if b[i] > b_max:
                b[i] = b_max
            elif b[i] < b_min:
                b[i] = b_min
        """

        # Evaluate L
        """
        mu_Y_new = find_sum(Y_new) / len(Y_new)
        mu_Y_t = find_sum(Y_t) / len(Y_t)
        """
        mu_Y_new = np.matmul(np.matmul(Y_new, np.linalg.inv(K_xx + lam * np.eye(len(K_xx)))), np.transpose(X_s))
        mu_Y_t = np.matmul(np.matmul(Y_t, np.linalg.inv(K_tt + lam * np.eye(len(K_tt)))), np.transpose(X_t))
        L = pow((mu_Y_new - mu_Y_t), 2) + pow((np.linalg.norm(w - 1)), 2) + pow(np.linalg.norm(b), 2)
       
        """
        print("w_old is")
        print(w_old)
        print(w)

        print("b_old is")
        print(b_old)
        print(b)
        """

        # Check if we can break
        # 2 criterion: 1) L is small enough 2) w, b converges
        # Break when either one of these satisfies
        if L < 0.000001:
            break
        if check_converge(w, b, w_old, b_old):
            print(num_itr)
            break

        # predict Y^tU using {X_s, Y_new} \cup {X_tL, Y_tL}
        # essnetially, update Y_new
        Y_new = find_y_new(X_s, Y_s, w, b)
        K_ynew = find_kernel(Y_new, Y_new)
    

    return w, b


if __name__ == "__main__":
    X_s, Y_s, X_t, Y_t = create_data()
    w, b = cdm(X_s, Y_s, X_t, Y_t)

    print(w)
    print(b)

    
    plt.plot(X_s, w, 'r*', X_s, b, 'bo')
    plt.show()
    
    prediction = np.zeros(len(X_s))
    for i in range(len(X_s) - 1):
        prediction[i] = Y_s[i] * w[i] * X_s[i] + b[i] * X_s[i]

    print(prediction)
    fig, ax = plt.subplots()
    ax.plot(X_s, prediction, 'go', label = 'prediction')
    ax.plot(X_t, Y_t, 'bo--', label = 'taget')
    ax.plot(X_s, Y_s, 'r*', label = 'source')
    legend = ax.legend(loc='lower left', shadow=False, fontsize='small')

    legend.get_frame().set_facecolor('#00FFCC')

    plt.show()

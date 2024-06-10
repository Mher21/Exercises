import pandas as pd
from scipy.sparse import csr_matrix
import time, numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('./ratings.csv',nrows=10000)
ratings = csr_matrix(
    (data.rating.values, (data.userId.values, data.movieId.values)),
    shape=(data.userId.max() + 1, data.movieId.max() + 1)
).toarray()

class MatrixFactorization():
    def __init__(self, ratings, n_factors=100, l_rate=0.01, alpha=0.01, n_iter=100):
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        print(self.n_users*self.n_items)
        self.non_zero_row_ind, self.non_zero_col_ind = ratings.nonzero()
        self.n_interac = len(ratings[np.where(ratings != 0)])
        self.ind_lst = list(range(self.n_interac))
        self.n_factors = n_factors
        self.l_rate = l_rate  # learning rate, Constant that multiplies the update term
        self.alpha = alpha  # lambda Constant that multiplies the regularization term
        self.n_iter = n_iter
        self.mse_lst = []
        self.wait = 10
        self.tol = 1e-3
        self.n_iter_no_change = 10
        self.verbose = True
        self.stop = False
        self.predicted_ratings = np.zeros_like(self.ratings)

    def initialize(self, ):
        self.now = time.time()
        # Initialize Bias Values
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        # initialize user & item vectors
        self.user_vecs = np.random.normal(scale=1 / self.n_factors, size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1 / self.n_factors, size=(self.n_items, self.n_factors))
        # compute global bias
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        self.evaluate_the_model(0)

    def predict(self, u, i):
        return self.global_bias + self.user_biases[u] + self.item_biases[i] + self.user_vecs[u] @ self.item_vecs[i]

    def update_biases_and_vectors(self, error, u, i):
        # Update biases
        # b_u = b_u + \eta * (e_{ui} - \lambda * b_u)
        self.user_biases[u] += self.l_rate * (error - self.alpha * self.user_biases[u])
         # b_i = b_i + \eta * (e_{ui} - \lambda * b_i)
        self.item_biases[i] += self.l_rate * (error - self.alpha * self.item_biases[i])
        # Update User and item Vectors
         # p_u = p_u + \eta * (e_{ui} * q_i - \lambda * p_u)
        self.user_vecs[u, :] += self.l_rate * (error * self.item_vecs[i, :] - self.alpha * self.user_vecs[u, :])
        # q_i = q_i + \eta * (e_{ui} * p_u - \lambda * q_i)
        self.item_vecs[i, :] += self.l_rate * (error * self.user_vecs[u, :] - self.alpha * self.item_vecs[i, :])

    def evaluate_the_model(self, epoch):
        tot_square_error = 0
        for index in self.ind_lst:
            # Extracting user item information indices in which we have a rating
            u, i = self.non_zero_row_ind[index], self.non_zero_col_ind[index]
            pred_rat = self.predict(u, i)
            tot_square_error += (self.ratings[u, i] - pred_rat) ** 2
        mse = tot_square_error / self.n_interac
        self.mse_lst.append(mse)
        if self.verbose:
            print(f"---> Epoch {epoch}")
            temp = np.round(time.time() - self.now, 3)
            print(f"ave mse {np.round(self.mse_lst[-1], 3)} ===> Total training time: {temp} seconds.")

    def early_stopping(self, epoch):
        if (self.mse_lst[-2] - self.mse_lst[-1]) <= self.tol:
            if self.wait == self.n_iter_no_change:
                temp = np.round(time.time() - self.now, 3)
                if self.verbose: print(f"Convergence after {epoch} epochs time took: {temp} seconds.")
                self.stop = True
                self.conv_epoch_num = epoch
            self.wait += 1
        else:
            self.wait = 0

    def fit(self, ):
        self.initialize()
        for epoch in range(1, self.n_iter):
            np.random.shuffle(self.ind_lst)
            if self.stop == False:
                for index in self.ind_lst:
                    # Extracting user item information indices in which we have a rating
                    u, i = self.non_zero_row_ind[index], self.non_zero_col_ind[index]
                    pred_rat = self.predict(u, i)
                    error = self.ratings[u, i] - pred_rat
                    self.update_biases_and_vectors(error, u, i)
                self.evaluate_the_model(epoch)
                self.early_stopping(epoch)
        
    def get_user_pred_ratings(self,user_ids) :
        print("Loading predicting ratings and recommendations...")
        recomms = {user_id: [] for user_id in user_ids}
        for u in range(self.n_users):
            for i in range(self.n_items):
                if self.ratings[u, i] == 0 and self.predicted_ratings[u, i] == 0:
                    self.predicted_ratings[u, i] = self.predict(u, i)
                if u in user_ids and self.predicted_ratings[u, i] >= 4.5:
                    recomms[u].append((i, self.predicted_ratings[u, i]))
        for user_id in user_ids:
            print(f"Recommendations for user_id = {user_id}: {recomms[user_id]}")
        
    def plot_the_score(self, ):
        plt.figure(figsize=(18, 6))
        plt.plot(range(1, 1 + len(self.mse_lst)), self.mse_lst, marker='o')
        plt.title("GD USER & ITEM vector's MSE loss vs epochs", fontsize=20)
        plt.xlabel('Number of epochs', fontsize=18)
        plt.ylabel('mean square error', fontsize=18)
        plt.xticks(range(1, self.conv_epoch_num + 5), fontsize=15, rotation=90)
        plt.yticks(np.linspace(min(self.mse_lst), max(self.mse_lst), 15), fontsize=15)
        plt.grid()
        plt.show()

obj = MatrixFactorization(ratings)
obj.fit()
obj.plot_the_score()
input_string = input("Enter user ids separated by commas,for receiving recommendations: ")
user_ids = list(map(int, input_string.split(',')))
obj.get_user_pred_ratings(user_ids)



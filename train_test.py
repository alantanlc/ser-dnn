#   PREPARE TRAIN SET AND TEST SET
#
#   In order to splt the dataset into train-test set, we are going to use scikit-learn's train_test_split
#   Here, we take 70% of the data for training and the remaining 30% for testing

X = np.load('X.npy')
y = np.load('y.npy')
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
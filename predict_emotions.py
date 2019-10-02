# If you look at the 'predict' variable you will see that instead of predicting emotions it is showing us some bunch of numbers.
# Now we will convert these numbers to corresponding emotion labels

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# predicted emotions from the test set
y_pred = np.argmx(predict, 1)
predicted_emo = []
for i in range(0, test_y,shape[0]):
    emo = emotions[y_pred[1]]
    predicted_emo.append(emo)
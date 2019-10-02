# Now that we have both the actual labels and predicted labels it's time to make a confusion matrix to get the overall picture of our emotion classifier's performance.

# generate the confusion matrix
cm = confusion_matrix(actual_emo, predicted_emo)
index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
cm_df = pd.DataFrame(cm, index, columns)
plt.figure(figsize=(10,6))
sns.heatmap(cm_df, annot=True)
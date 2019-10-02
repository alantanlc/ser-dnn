#   The next step is to call the functions with proper directories.
#   We are going to save all our audio features to the variable 'X'
#   and all the labels to variable 'y'.

#   FILE NAMING CONVENTION
#
#   Each of the 7356 RAVDESS files has a unique filename.
#   The filename consists of a 7-part numerical identifier (e.g. 02-01-06-01-02-01-12.mp4)
#   These identifiers define the stimulus characteristics:
#
#   FILENAME IDENTIFIERS
#
#   Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
#   Vocal channel (01 = speech, 02 = song)
#   Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
#   Emotional intensity (01 = normal, 02 = strong). Note: There is no strong intensity for the 'neutral' emotion
#   Statement (01 = 'Kids are talking by the door', 02 = 'Dogs are sitting by the door')
#   Repetition (01 = 1st repetition, 02 = 2nd repetition)
#   Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)
#   
#   FILENAME EXAMPLE: 02-01-06-01-02-01-12.mp4
#   
#   1. Video-only (02)
#   2. Speech (01)
#   3. Fearful (06)
#   4. Normal intensity (01)
#   5. Statement 'dogs' (02)
#   6. 1st Repetition (01)
#   7. 12th Actor (12)
#   8. Female, as the actor ID number is even

#   change the main_dir accordingly...
main_dir = './Audio_Speech_Actors_01-24'
sub_dir = os.listdir(main_dir)

print('\ncollecting features and labels...')
print('\nthis will take some time...')
features, labels = parse_audio_files(main_dir, sub_dir)
print('done')

np.save('X', features)

# one hot encoding labels
labels = one_hot_encode(labels)
np.save('y', labels)

print('\nEnd of program')
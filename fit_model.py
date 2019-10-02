# create the model
model = create_model()
# train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=4)
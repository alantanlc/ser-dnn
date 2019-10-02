#   BUILDING THE DNN
#
#   Our deep neural network consists of one input layer, three hidden layers, and one output layer. This architecture was randomly chosen (you may try a different one). 20% dropout rate is used to prevent over-fitting during the training phase.
#
#   If you do not know how to define a DNN then keras documentation is always there to help you.

# dnn parameters
n_dim = train_x.shape[1]
n_classes = train_y.shape[1]
n_hidden_units_1 = n_dim
n_hidden_units_2 = 400 # approx n_dim * 2
n_hidden_units_3 = 200 # half of layer 2
n_hidden_units_4 = 100

# defining the mode
def create_model(activation_function='relu', init_type='normal', optimiser='adam', dropout_rate=0.2):
    model = Sequential()
    # layer 1
    model.add(Dense(n_hidden_units_1, input_dim=n_dim, init=init_type, activation=activation_function))
    # layer 2
    model.add(Dense(n_hidden_units_2, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer 3
    model.add(Dense(n_hidden_units_3, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer 4
    model.add(Dense(n_hidden_units_4, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # output layer
    model.add(Dense(n_classes, init=init_type, activation='softmax'))
    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    return model
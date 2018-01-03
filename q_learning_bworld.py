import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import rl_blocksworld
import keras
from rl_blocksworld import * 
from utilities import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


epsilon = 0.3 #exploration factor
actions = [UP, DOWN, LEFT, RIGHT]

#TODO: Generate Random maze
initial = np.arange(16).reshape((4,4))
initial[:] = 0
initial[3,0] = A_MARKER
initial[3,1] = B_MARKER
initial[3,2] = C_MARKER
initial[3,3] = AGENT_MARKER

goal = np.arange(16).reshape((4,4))
goal[:] = 0
goal[1,1] = A_MARKER
goal[2,1] = B_MARKER
goal[3,1] = C_MARKER

plt.grid('on')
ax = plt.gca()
nrows, ncols = initial.shape
ax.set_xticks(np.arange(0.5, nrows, 1))
ax.set_yticks(np.arange(0.5, ncols, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.ion()

blocksworld = BlocksWorld(initial, goal, agent_location=(3,3))

def test_random_puzzle(count):
    for i in range(count):
        blocksworld.reset()
        print(blocksworld.puzzle)
        print("***" * 5)

def show(world): 
    canvas = np.copy(world.puzzle)
    canvas = world.draw_env(canvas)
    plt.imshow(canvas, interpolation='none', cmap='tab20c')
    plt.show(block=False)
    plt.pause(0.2)
    return canvas


### Random Movements
def brownian_motion():
    show(blocksworld)
    for count in range(10000):
        action = blocksworld.random_valid_action()
        canvas, reward, game_status = blocksworld.act( action )
        print("Step {:}: Reward = {:}, Average = {:}".format(count+1, reward,blocksworld.total_reward / (count+1)))
        show(blocksworld)
        if game_status == rl_blocksworld._COMPLETE_:
            break

    canvas = canvas.reshape((4,4))
    print("Total Reward = " + str(blocksworld.total_reward))
    print(blocksworld.puzzle)
    plt.imshow(canvas, interpolation='none', cmap='tab20c')
    plt.show(block=True)


def build_model(shape,num_actions):
    model = Sequential()
    
    model.add(Dense(100, input_shape=shape, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_actions, activation='softmax'))
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy') 
    model.compile(loss='mse',optimizer=Adam(lr=0.01))
    
    # model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=shape))
    # model.add(Conv2D(64, (2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 1)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_actions, activation='softmax'))

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #             optimizer=keras.optimizers.Adadelta(),
    #             metrics=['accuracy'])
    return model


def qtrain(model, world, **opt):
    global epsilon, actions
    epochs = opt.get('epochs', 15000)
    max_memory = opt.get('max_memory', 1000)
    batch_size = opt.get('batch_size', 100)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)
    
    experience = Experience(model, max_memory=max_memory)
    success_history = []
    win_rate = 0.0

    for epoch in range(epochs):
        loss = 0.0
        blocksworld.reset()
        envstate = blocksworld.observe()
        game_status = rl_blocksworld._IN_PROGRESS_
        game_over = False
        n_episodes = 0
        # show(blocksworld)
        while not game_over and ( n_episodes < 25):
            prev_envstate = envstate
            if np.random.rand() < epsilon:
                action = blocksworld.random_valid_action()
            else:
                action = np.argmax(model.predict(prev_envstate)[0])

            envstate, reward, game_status = blocksworld.act(action)
            # show(blocksworld)
            if game_status == rl_blocksworld._COMPLETE_:
                success_history.append(1)
                game_over = True 
            elif game_status == rl_blocksworld._TERMINATE_:
                success_history.append(0)
                game_over = True 
            else:
                game_over = False

            episode = (prev_envstate, action, reward, envstate, game_status)
            experience.remember(episode)
            n_episodes += 1
            # print("Step {:}: Reward = {:}, Average = {:}, Total = {:}".format(n_episodes, reward,blocksworld.total_reward / (n_episodes), blocksworld.total_reward))

            inputs, targets = experience.get_data(batch_size=batch_size)
            # loss = model.train_on_batch(inputs,targets)
            model.fit(inputs,targets,epochs=8,batch_size=16,verbose=0,)
            loss = model.evaluate(inputs, targets, verbose=0)
        win_rate = sum(success_history) / len(success_history)
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | Average Reward: {:2f} | time: {}"
        print(template.format(epoch, epochs-1, loss, n_episodes, sum(success_history), win_rate, blocksworld.total_reward / (n_episodes), t))
        if win_rate > 0.9:
            epsilon = 0.05 
        if win_rate > 0.95:
            break
        # if sum(success_history[-10:]) == 10
    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, batch_size, t))
    return seconds


def play_game(model, blocksworld, agent_location=None):
    blocksworld.reset()
    envstate = blocksworld.observe()
    show(blocksworld)
    while True:
        prev_envstate = envstate
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        envstate, reward, game_status = blocksworld.act(action)
        if game_status == rl_blocksworld._COMPLETE_:
            return True
        elif game_status == rl_blocksworld._TERMINATE_:
            return False

# shape = (blocksworld.puzzle.shape[0],blocksworld.puzzle.shape[1],1)
shape = (blocksworld.puzzle.size,)
model = build_model(shape,len(actions))
qtrain(model, blocksworld, epochs=10, max_memory=8*blocksworld.puzzle.size)
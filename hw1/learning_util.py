import pickle
import numpy as np
import tensorflow as tf
import tf_util

from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split


def build_model(config_dict, input_shape, output_shape):
    # creating feed forward neural network
    model = Sequential()
    model.add(Dense(units=config_dict['input_layer']['units'],
                    activation=config_dict['input_layer']['activation'],
                    input_shape=input_shape))

    for layer in config_dict['hidden_layers']:
        model.add(Dense(units=config_dict['hidden_layers'][layer]['units'],
                        activation=config_dict['hidden_layers'][layer]['activation']))
    model.add(Dense(units=output_shape,
                    activation=config_dict['output_layer']['activation']))

    model.compile(loss=config_dict['optimize']['loss'],
                  optimizer=config_dict['optimize']['optimizer'],
                  metrics=config_dict['optimize']['metrics'])

    # log TODO
    print(model.summary())

    return model


def get_train_test_data(x, y, test_size):
    x_train, x_test, \
    y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

    return x_train, y_train, x_test, y_test


def load_data(data_file, test_size=0.33):
    # open data file
    with open(data_file, 'rb') as f:
        data = pickle.loads(f.read())

    # extract observations, and actions
    observations_data = np.array(data['observations'])
    actions_data = np.array(data['actions'])

    actions_data = actions_data.reshape(actions_data.shape[0], actions_data.shape[2])

    # train test split
    return get_train_test_data(observations_data, actions_data, test_size)


def fit_model(config_dict, model,
              x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train,
              batch_size=config_dict['train']['batch_size'],
              epochs=config_dict['train']['epochs'],
              verbose=config_dict['train']['verbose'])
    score = model.evaluate(x_test, y_test, verbose=0)

    return model, score


def final_env_test(args, cfg, method='behavioral_cloning'):
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        # load model
        model = load_model(cfg.MODELS.MODELS_DIR[0][method]['dir'] + '/' +
                           args.envname + cfg.MODELS.MODELS_DIR[0][method]['suffix'], )

        returns = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = model.predict(obs[None, :])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    # TODO
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        # TODO
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

import load_policy
import numpy as np
import tensorflow as tf
import tf_util
from config import get_cfg_defaults
from learning_util import load_data, get_train_test_data, \
    build_model, fit_model, final_env_test


def train(args, cfg, save=True):
    with tf.Session():
        tf_util.initialize()

        # load data
        x_train, y_train, x_test, y_test = load_data(args.data_file,
                                                     cfg.DATA.TEST_TRAIN_RATIO)
        # get model dict, and build model
        model_dict = cfg.MODELS.ARCH_DCIT[0]['dagger']
        model = build_model(config_dict=model_dict,
                            input_shape=(x_train.shape[1],),
                            output_shape=y_train.shape[1])

        # load optimum policy
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        # get gym env
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        # start DAgger
        mean_rewards = []
        stds = []
        for iter in range(cfg.MODELS.DAGGER_LOOPS):
            # TODO add logger
            print("Dagger iter: {}".format(iter))

            # I) train model on data
            model, score = fit_model(model_dict,
                                     model,
                                     x_train, y_train,
                                     x_test, y_test)
            # TODO logger
            print("Evaluation of model on test data is {}".format(score))
            # II) run policy simulation
            # III) expert would label data too

            new_observations = []
            new_expert_actions = []
            returns = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0

                while not done:
                    expert_action = policy_fn(obs[None, :])
                    # get new data
                    new_expert_actions.append(expert_action)
                    new_observations.append(obs)

                    action = model.predict(obs[None, :])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1

                    if steps % 100 == 0:
                        # TODO
                        print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            mean_rewards.append(np.mean(returns))
            stds.append(np.std(returns))

            # aggregate data
            new_observations = np.array(new_observations)
            new_expert_actions = np.array(new_expert_actions)
            new_expert_actions = new_expert_actions.reshape(new_expert_actions.shape[0],
                                                            new_expert_actions.shape[2])

            observations_data = np.concatenate((x_train, x_test, new_observations))
            actions_data = np.concatenate((y_train, y_test, new_expert_actions))

            x_train, y_train, \
            x_test, y_test = get_train_test_data(observations_data,
                                                 actions_data,
                                                 cfg.DATA.TEST_TRAIN_RATIO)

            # TODO logger
            print(mean_rewards)
            print(stds)

            if iter + 1 == cfg.MODELS.DAGGER_LOOPS and save:
                model.save(cfg.MODELS.MODELS_DIR[0]['dagger']['dir'] + '/' +
                           args.envname + cfg.MODELS.MODELS_DIR[0]['dagger']['suffix'])


def test(args, cfg):
    final_env_test(args, cfg, method='dagger')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.freeze()

    if args.test:
        test(args, cfg)
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()

from config import get_cfg_defaults
from learning_util import load_data, build_model, \
    fit_model,final_env_test


def train(cfg, data_file, env_name, save=True):
    # load data
    x_train, y_train, x_test, y_test = load_data(data_file, cfg.DATA.TEST_TRAIN_RATIO)

    # build model
    model_dict = cfg.MODELS.ARCH_DCIT[0]['behavioral_cloning']
    model = build_model(config_dict=model_dict,
                        input_shape=(x_train.shape[1],),
                        output_shape=y_train.shape[1])

    # train, and test the model
    model, score = fit_model(model_dict,
                             model,
                             x_train, y_train,
                             x_test, y_test)

    # TODO
    print("Score of the model {}".format(score))

    # save trained model
    if save:
        model.save(cfg.MODELS.MODELS_DIR[0]['behavioral_cloning']['dir'] + '/' +
                   env_name + cfg.MODELS.MODELS_DIR[0]['behavioral_cloning']['suffix'])

    return model


def test(args, cfg):
    final_env_test(args, cfg, method='behavioral_cloning')


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
        train(cfg, args.data_file, args.envname)


if __name__ == "__main__":
    main()

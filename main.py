import argparse
from runner import Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False, help='train or evaluate, default eval')
    parser.add_argument('--mpath', type=str, default=None, help='eval model path')
    # some other parser
    args = parser.parse_args()
    runner = Runner()
    if args.train:
        runner.train()
    else:
        if args.mpath is None:
            args.mpath = "saved/PongNoFrameskip-v4_DQN/models"
        runner.eval(model_path=args.mpath)
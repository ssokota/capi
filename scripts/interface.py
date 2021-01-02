import argparse
import os

import numpy as np
import torch
from torch.optim import Adam

from capi.agents import Agent
from capi.games import Game
from capi.models import NN
from capi.trainers import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=12)
    parser.add_argument("--num_utterances", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--epsilon", type=float, default=1 / 10)
    parser.add_argument("--policy_weight", type=float, default=1 / 100)
    parser.add_argument("--num_episodes", type=int, default=2_000)
    parser.add_argument("--write_every", type=int, default=10)
    parser.add_argument("--directory", type=str, default="results")
    parser.add_argument("--jobnum", type=int, default=0)
    parser.add_argument("--strict_type_check", type=bool, default=False)
    args = parser.parse_args()
    if args.strict_type_check:
        os.environ["strict_type_check"] = "1"
    else:
        os.environ["strict_type_check"] = "0"
    input_size = 2 * args.num_items + 2 * args.num_utterances
    nn = NN(input_size, args.hidden_size, args.num_items, args.num_utterances)
    opt = Adam(nn.parameters(), lr=args.lr)
    g = Game(args.num_items, args.num_utterances)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        args.num_items,
        args.num_utterances,
        nn,
        opt,
        args.num_samples,
        args.epsilon,
        args.policy_weight,
        device
    )
    trainer = Trainer(g, agent, args.directory, args.jobnum)
    trainer.run(args.num_episodes, args.write_every)

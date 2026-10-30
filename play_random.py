import os
import random

import numpy as np
import torch

from main import Connect4Env, Connect4CNN, tactical_move

MODEL_PATH = "connect4_cnn.pth"
NUM_GAMES = 1000


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find '{MODEL_PATH}'. Run main.py to train a model first."
        )
    model = Connect4CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # switches BatchNorm to inference mode
    return model


def ai_move(model, env, device, use_tactics=True):
    """Red picks the highest-rated column, Yellow picks the lowest."""
    if use_tactics:
        forced = tactical_move(env)
        if forced is not None:
            return forced

    with torch.no_grad():
        x = env.get_tensor_input().unsqueeze(0).to(device)
        q = model(x)[0].cpu().numpy()

    valid = env.get_valid_moves()
    if env.current_player == 1:
        best = max(q[c] for c in valid)
    else:
        best = min(q[c] for c in valid)

    best_moves = [c for c in valid if np.isclose(q[c], best)]
    return random.choice(best_moves)


def random_move(env):
    """Pick a random valid column."""
    return random.choice(env.get_valid_moves())


def play_game(model, device, ai_color):
    """Play one game. ai_color is 1 (Red, moves first) or -1 (Yellow).
    Returns the winner: 1, -1, or 0 for a draw."""
    env = Connect4Env()
    winner = None

    while winner is None:
        if env.current_player == ai_color:
            col = ai_move(model, env, device)
        else:
            col = random_move(env)
        winner = env.make_move(col)

    return winner


def main():
    device = get_device()
    model = load_model(device)
    print(f"Loaded '{MODEL_PATH}' on {device}.")

    # results[ai_color] = [AI wins, random wins, draws]
    results = {1: [0, 0, 0], -1: [0, 0, 0]}

    for i in range(NUM_GAMES):
        ai_color = 1 if i % 2 == 0 else -1  # alternate so AI plays both colors equally
        winner = play_game(model, device, ai_color)

        if winner == ai_color:
            results[ai_color][0] += 1
        elif winner == 0:
            results[ai_color][2] += 1
        else:
            results[ai_color][1] += 1

    names = {1: "Red (first)", -1: "Yellow (second)"}
    total = [sum(results[c][k] for c in results) for k in range(3)]

    print(f"\nResults over {NUM_GAMES} games vs a random player:")
    for color in (1, -1):
        w, l, d = results[color]
        print(f"  AI as {names[color]:16} | Wins: {w} | Losses: {l} | Draws: {d}")
    print(
        f"  Overall: Wins {total[0]} ({total[0] / NUM_GAMES:.1%}) | "
        f"Losses {total[1]} | Draws {total[2]}"
    )


if __name__ == "__main__":
    main()
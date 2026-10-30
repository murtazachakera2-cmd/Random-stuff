
import os
import random
 
import numpy as np
import torch
 
from main import Connect4Env, Connect4CNN, tactical_move

MODEL_PATH = "connect4_cnn.pth"
SYMBOLS = {0: ".", 1: "R", -1: "Y"}
 
 
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
    model.eval()  # important: switches BatchNorm to inference mode
    return model
 
 
def print_board(env):
    print()
    print(" " + " ".join(str(c + 1) for c in range(env.cols)))
    for r in range(env.rows):
        print(" " + " ".join(SYMBOLS[env.board[r][c]] for c in range(env.cols)))
    print()
 
 
def ai_move(model, env, device, use_tactics=True):
    """Red picks the highest-rated column, Yellow picks the lowest."""
    if use_tactics:
        forced = tactical_move(env)
        if forced is not None:
            return forced

    # ... the rest stays exactly as it is
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
 
 
def human_move(env):
    valid = env.get_valid_moves()
    while True:
        raw = input(f"Your move (columns {[c + 1 for c in valid]}): ").strip()
        if raw.isdigit() and (int(raw) - 1) in valid:
            return int(raw) - 1
        print("Invalid move, try again.")
 
 
def choose_side():
    while True:
        choice = input("Play as Red (moves first) or Yellow? [r/y]: ").strip().lower()
        if choice in ("r", "red"):
            return 1
        if choice in ("y", "yellow"):
            return -1
        print("Please enter r or y.")
 
 
def play_game(model, device):
    human = choose_side()
    env = Connect4Env()  # Red (1) always moves first
    winner = None
 
    print_board(env)
 
    while winner is None:
        if env.current_player == human:
            col = human_move(env)
        else:
            col = ai_move(model, env, device)
            print(f"AI plays column {col + 1}")
 
        winner = env.make_move(col)
        print_board(env)
 
    if winner == 0:
        print("It's a draw!")
    elif winner == human:
        print("You win!")
    else:
        print("The AI wins!")
 
 
def main():
    device = get_device()
    model = load_model(device)
    print(f"Loaded '{MODEL_PATH}' on {device}.")
 
    while True:
        play_game(model, device)
        if input("Play again? [y/n]: ").strip().lower() not in ("y", "yes"):
            break
 
 
if __name__ == "__main__":
    main()
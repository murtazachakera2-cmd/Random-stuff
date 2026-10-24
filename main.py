import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os


# 1. CONNECT 4 ENVIRONMENT
class Connect4Env:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((6, 7), dtype=int)  # 0: Empty, 1: Red, -1: Yellow
        self.current_player = 1                   # 1: Red, -1: Yellow

    def clone(self):
        env = Connect4Env()
        env.board = np.copy(self.board)
        env.current_player = self.current_player
        return env

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def make_move(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.current_player
                break
        winner = self.check_winner()
        self.current_player *= -1
        return winner

    def check_winner(self):
        # Check horizontal, vertical, and diagonal lines for 4-in-a-row
        board = self.board
        for r in range(self.rows):
            for c in range(self.cols):
                p = board[r][c]
                if p == 0:
                    continue
                # Horizontal
                if c + 3 < self.cols and all(board[r][c+i] == p for i in range(4)):
                    return p
                # Vertical
                if r + 3 < self.rows and all(board[r+i][c] == p for i in range(4)):
                    return p
                # Positive Diagonal
                if r + 3 < self.rows and c + 3 < self.cols and all(board[r+i][c+i] == p for i in range(4)):
                    return p
                # Negative Diagonal
                if r - 3 >= 0 and c + 3 < self.cols and all(board[r-i][c+i] == p for i in range(4)):
                    return p
        if len(self.get_valid_moves()) == 0:
            return 0  # Draw
        return None  # Game ongoing

    def get_tensor_input(self):
        # Shape: (3, 6, 7)
        plane_red = (self.board == 1).astype(np.float32)
        plane_yellow = (self.board == -1).astype(np.float32)
        plane_turn = np.full((6, 7), 1.0 if self.current_player == 1 else 0.0, dtype=np.float32)
        return torch.tensor(np.stack([plane_red, plane_yellow, plane_turn]))



# 2. CONVOLUTIONAL NEURAL NETWORK
class Connect4CNN(nn.Module):
    def __init__(self):
        super(Connect4CNN, self).__init__()
        
        # Convolutional layers to process 2D spatial board layout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        
        # Fully connected layers to evaluate all 7 columns
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 7)
        self.tanh = nn.Tanh()  # Output range: [-1, 1]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        out = self.tanh(self.fc2(x))  # Range: -1 (Yellow win) to +1 (Red win)
        return out


## MCTS class
class MCTSNode:
    def __init__(self, env, parent=None, move=None):
        self.env = env
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.env.get_valid_moves())

    def best_child(self, c_param=1.414):
        choices_weights = [
            (child.value_sum / child.visits) * self.env.current_player
            + c_param * np.sqrt(
                2 * np.log(self.visits) / child.visits
            )
            for child in self.children.values()
        ]

        return list(self.children.values())[np.argmax(choices_weights)]


def mcts_search(root_env, num_simulations=50):
    """
    Performs MCTS simulations from the given environment position.
    Returns target Q-values for each column from the perspective of Red (+1) and Yellow (-1).
    """
    root = MCTSNode(root_env.clone())

    for _ in range(num_simulations):
        node = root
        sim_env = root_env.clone()

        # 1. Selection
        while node.is_fully_expanded() and len(node.children) > 0:
            node = node.best_child()
            sim_env.make_move(node.move)

        # 2. Expansion
        winner = sim_env.check_winner()
        if winner is None:
            valid_moves = sim_env.get_valid_moves()
            unvisited_moves = [m for m in valid_moves if m not in node.children]
            if unvisited_moves:
                move = random.choice(unvisited_moves)
                winner = sim_env.make_move(move)
                child_node = MCTSNode(sim_env.clone(), parent=node, move=move)
                node.children[move] = child_node
                node = child_node

        # 3. Rollout / Simulation (Play randomly until game ends)
        while winner is None:
            valid_moves = sim_env.get_valid_moves()
            if not valid_moves:
                winner = 0
                break
            move = random.choice(valid_moves)
            winner = sim_env.make_move(move)

        # 4. Backpropagation (Adjusted for current player's perspective)
        absolute_reward = float(winner)  # 1 for Red win, -1 for Yellow win, 0 for Draw
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += absolute_reward
            curr = curr.parent

    # Compute target Q-values for all 7 actions from root's perspective
    action_values = np.zeros(7, dtype=np.float32)
    
    for col in range(7):
        if col in root.children and root.children[col].visits > 0:
            # Average outcome estimated by MCTS from root's perspective
            action_values[col] = root.children[col].value_sum / root.children[col].visits
        else:
            # Mask invalid moves or unvisited options with extreme loss penalty
            action_values[col] = -1.0

    return action_values

def play_one_game(mcts_simulations=40):
    env = Connect4Env()

    states = []
    targets = []
    masks = []
    chosen_moves = []

    winner = None

    # Training loop
    while winner is None:

        valid_moves = env.get_valid_moves()

        # Save the current board before making a move
        state = env.get_tensor_input()

        # MCTS estimates the value of each column
        q_values = mcts_search(
            env,
            num_simulations=mcts_simulations
        )

        # Remember which columns are legal
        mask = np.zeros(7, dtype=bool)
        mask[valid_moves] = True

        # Red wants the highest score.
        # Yellow wants the lowest score.
        if env.current_player == 1:
            best_score = max(q_values[c] for c in valid_moves)
        else:
            best_score = min(q_values[c] for c in valid_moves)

        # Randomly choose between equally rated moves
        best_moves = [
            c for c in valid_moves
            if np.isclose(q_values[c], best_score)
        ]

        move = random.choice(best_moves)

        # Record the information for training
        states.append(state)
        targets.append(q_values.copy())
        masks.append(mask)
        chosen_moves.append(move)

        # Actually make the move on the board
        winner = env.make_move(move)

    # The game has ended.
    # +1 = Red wins, -1 = Yellow wins, 0 = draw
    game_result = float(winner)

    # Replace the MCTS estimate for each move actually
    # played with the result of the completed game.
    for target, move in zip(targets, chosen_moves):
        target[move] = game_result

    return (
        torch.stack(states),
        torch.tensor(np.stack(targets), dtype=torch.float32),
        torch.tensor(np.stack(masks), dtype=torch.bool),
        winner
    )

#TRAINING LOOP
def train():
    losses = []
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Connect4CNN().to(device)

    model_path = "connect4_cnn.pth"

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        print("Loaded existing model. Continuing training...")
    else:
        print("Starting with a fresh model...")

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    num_games = 100
    mcts_simulations = 300

    print(f"Training on {device}...")
    print(f"Total games: {num_games}")

    red_wins = 0
    yellow_wins = 0
    draws = 0
    game_number = 1
    with open("data.txt","a") as file:
        while True:

            if (game_number > num_games):
                break


            # STEP 1: PLAY ONE COMPLETE GAME
            states, targets, masks, winner = play_one_game(
                mcts_simulations=mcts_simulations
            )

            # Move the entire game's data to the device
            states = states.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            # STEP 2: MAKE PREDICTIONS FOR EVERY POSITION
            optimizer.zero_grad()

            predictions = model(states)

            # Calculate loss only for legal moves
            loss = criterion(
                predictions[masks],
                targets[masks]
            )

            # STEP 3: UPDATE THE CNN ONCE FOR THIS GAME
            loss.backward()
            optimizer.step()

            # Count the result
            if winner == 1:
                red_wins += 1
                result = "Red won"
            elif winner == -1:
                yellow_wins += 1
                result = "Yellow won"
            else:
                draws += 1
                result = "Draw"

            file.write(
                f"Game {game_number}/{num_games} | "
                f"Moves: {len(states)} | "
                f"Result: {result} | "
                f"Loss: {loss.item():.4f}\n"
            )
            print(
                f"Game {game_number}/{num_games} | "
                f"Moves: {len(states)} | "
                f"Result: {result} | "
                f"Loss: {loss.item():.4f}"
            )
            losses.append(loss.item())

            # Save progress every 10 games
            if game_number % 10 == 0:
                torch.save(model.state_dict(), model_path)

            game_number += 1

        # Final save
        torch.save(model.state_dict(), model_path)

        file.write("\nTraining complete!")
        file.write(f"Red wins: {red_wins}\n")
        file.write(f"Yellow wins: {yellow_wins}\n")
        file.write(f"Draws: {draws}\n")
        file.write(f"Model saved to '{model_path}'.\n\n")

        print("Training complete!\n")
        print(f"Red wins: {red_wins}")
        print(f"Yellow wins: {yellow_wins}")
        print(f"Draws: {draws}")
        print(f"Model saved to '{model_path}'.")
        print(f"Average loss: {sum(losses) / len(losses)}")
        print(f"Maximum loss: {max(losses)}")


if __name__ == "__main__":
    train()
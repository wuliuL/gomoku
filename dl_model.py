#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Training Module for Gomoku
Supports self-play data collection and model training with checkpointing
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# ============== Constants ==============
BOARD_SIZE = 15
EMPTY = 0
BLACK_STONE = 1
WHITE_STONE = 2

# ============== Neural Network ==============
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class GomokuNet(torch.nn.Module):
    """
    AlphaZero-style Gomoku Neural Network
    Input: 4 x 15 x 15 (current board + history)
    Output: Policy (225) + Value (1)
    """
    def __init__(self, board_size=15, channels=128, num_blocks=10):
        super().__init__()
        self.board_size = board_size
        
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU()
        )
        
        self.res_tower = torch.nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        
        self.policy_conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 2, 1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU()
        )
        self.policy_fc = torch.nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        self.value_conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 1, 1, bias=False),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        self.value_fc = torch.nn.Sequential(
            torch.nn.Linear(board_size * board_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        
        p = self.policy_conv(x).view(x.size(0), -1)
        policy = self.policy_fc(p)
        
        v = self.value_conv(x).view(x.size(0), -1)
        value = self.value_fc(v)
        
        return policy, value
    
    def get_action(self, state, valid_moves=None, temperature=0.5):
        """Get action from neural network"""
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            
            policy_logits, _ = self(state)
            policy = F.softmax(policy_logits, dim=-1)[0].cpu().numpy()
            
            if valid_moves:
                mask = np.zeros_like(policy)
                for r, c in valid_moves:
                    mask[r * self.board_size + c] = 1.0
                policy = policy * mask
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy = policy / policy_sum
                else:
                    policy = mask / len(valid_moves)
            
            idx = np.random.choice(self.board_size * self.board_size, p=policy)
            return (idx // self.board_size, idx % self.board_size)


# ============== Dataset ==============
class SelfPlayDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return (torch.FloatTensor(state),
                torch.FloatTensor(policy),
                torch.tensor(value, dtype=torch.float32))


# ============== Self-Play Trainer ==============
class SelfPlayTrainer:
    """Self-play trainer for Gomoku AI with cumulative training support"""
    
    def __init__(self, save_dir='models', board_size=15, channels=128, num_blocks=10):
        self.save_dir = save_dir
        self.board_size = board_size
        self.channels = channels
        self.num_blocks = num_blocks
        
        # Create model
        self.model = GomokuNet(board_size=board_size, channels=channels, num_blocks=num_blocks)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # Training stats
        self.game_count = 0
        self.win_count = {BLACK_STONE: 0, WHITE_STONE: 0}
        self.total_games = 0
        
        # Self-play data buffer
        self.data_buffer = []
        
        # Load existing model and stats if available
        self._load_checkpoint()
        
        # Create models directory
        os.makedirs(save_dir, exist_ok=True)
    
    def _load_checkpoint(self):
        """Load existing model and training progress"""
        # Try different checkpoint paths
        checkpoint_paths = [
            os.path.join(self.save_dir, 'gomoku_checkpoint.pth'),
            os.path.join(self.save_dir, 'gomoku_model.pth'),
            os.path.join(self.save_dir, 'gomoku_net_final.pt'),
        ]
        
        for path in checkpoint_paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    
                    # Load model weights
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    # Load training stats
                    if 'game_count' in checkpoint:
                        self.game_count = checkpoint['game_count']
                    if 'win_count' in checkpoint:
                        self.win_count = checkpoint['win_count']
                    if 'total_games' in checkpoint:
                        self.total_games = checkpoint['total_games']
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    print(f"Loaded checkpoint from {path}: {self.game_count} games, {self.total_games} total training games")
                    return True
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        
        print("No existing checkpoint found, starting fresh training")
        return False
    
    def save_checkpoint(self, filename='gomoku_checkpoint.pth'):
        """Save model checkpoint with training progress"""
        path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'game_count': self.game_count,
            'win_count': self.win_count,
            'total_games': self.total_games,
            'channels': self.channels,
            'num_blocks': self.num_blocks,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def save_model(self, filename='gomoku_model.pth'):
        """Save model only (for inference)"""
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def _get_state(self, board, player):
        """Get neural network input state"""
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        opponent = WHITE_STONE if player == BLACK_STONE else BLACK_STONE
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    state[0, i, j] = 1.0
                elif board[i][j] == opponent:
                    state[1, i, j] = 1.0
        
        state[2] = 1.0 if player == BLACK_STONE else 0.0
        return state
    
    def _get_candidates(self, board):
        """Get candidate move positions"""
        candidates = set()
        has_stones = False
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] != EMPTY:
                    has_stones = True
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                                if board[ni][nj] == EMPTY:
                                    candidates.add((ni, nj))
        
        if not has_stones:
            return [(7, 7)]
        
        if not candidates:
            candidates.add((7, 7))
        
        return list(candidates)
    
    def _check_win(self, board, i, j, player):
        """Check if player has won"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            x, y = i + dx, j + dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x][y] == player:
                count += 1
                x, y = x + dx, y + dy
            x, y = i - dx, j - dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x][y] == player:
                count += 1
                x, y = x - dx, y - dy
            if count >= 5:
                return True
        return False
    
    def _get_policy_target(self, board, player, valid_moves):
        """Generate policy target based on evaluation"""
        policy = np.zeros(self.board_size * self.board_size)
        
        for i, j in valid_moves:
            # Quick win check
            board[i][j] = player
            if self._check_win(board, i, j, player):
                board[i][j] = EMPTY
                policy[i * self.board_size + j] = 1.0
                break
            board[i][j] = EMPTY
            
            # Block opponent win
            enemy = WHITE_STONE if player == BLACK_STONE else BLACK_STONE
            board[i][j] = enemy
            if self._check_win(board, i, j, enemy):
                board[i][j] = EMPTY
                policy[i * self.board_size + j] = 0.8
                continue
            board[i][j] = EMPTY
        
        # Normalize
        total = policy.sum()
        if total > 0:
            policy = policy / total
        else:
            for i, j in valid_moves[:5]:
                policy[i * self.board_size + j] = 1.0 / len(valid_moves[:5])
        
        return policy
    
    def _get_value_target(self, board, player, winner):
        """Get value target based on game outcome"""
        if winner == player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0
    
    def self_play_game(self, verbose=False):
        """Play one self-play game and collect data"""
        board = [[EMPTY] * self.board_size for _ in range(self.board_size)]
        current_player = BLACK_STONE
        game_data = []
        move_count = 0
        
        while move_count < self.board_size * self.board_size:
            # Get state
            state = self._get_state(board, current_player)
            
            # Get valid moves
            valid_moves = self._get_candidates(board)
            
            # Get policy target
            policy = self._get_policy_target(board, current_player, valid_moves)
            
            # Store state and policy
            game_data.append((state.copy(), policy, current_player))
            
            # Get move from model (with some exploration)
            if random.random() < 0.3:  # 30% random exploration
                move = random.choice(valid_moves)
            else:
                move = self.model.get_action(state, valid_moves, temperature=0.5)
            
            i, j = move
            if board[i][j] != EMPTY:
                move = random.choice(valid_moves)
                i, j = move
            
            board[i][j] = current_player
            move_count += 1
            
            if verbose and move_count % 10 == 0:
                print(f"  Move {move_count}: ({i}, {j})")
            
            # Check win
            if self._check_win(board, i, j, current_player):
                winner = current_player
                break
            
            current_player = WHITE_STONE if current_player == BLACK_STONE else BLACK_STONE
        else:
            winner = 0  # Draw
        
        # Generate value targets for all positions
        final_data = []
        for state, policy, player in game_data:
            value = self._get_value_target(board, player, winner)
            final_data.append((state, policy, value))
        
        return final_data, winner
    
    def collect_games(self, num_games=10, verbose=False):
        """Collect self-play games"""
        for i in range(num_games):
            game_data, winner = self.self_play_game(verbose=verbose)
            self.data_buffer.extend(game_data)
            self.game_count += 1
            self.win_count[winner] += 1
            self.total_games += 1
            
            if verbose:
                winner_name = 'Black' if winner == BLACK_STONE else ('White' if winner == WHITE_STONE else 'Draw')
                print(f"Game {self.game_count}: {winner_name} wins")
        
        # Limit buffer size to prevent memory issues
        if len(self.data_buffer) > 100000:
            self.data_buffer = self.data_buffer[-80000:]
        
        return len(self.data_buffer)
    
    def train(self, epochs=1, batch_size=64, verbose=True):
        """Train the model on collected data"""
        if len(self.data_buffer) < 32:
            if verbose:
                print("Not enough data for training, collecting games first...")
            self.collect_games(5, verbose=False)
        
        if len(self.data_buffer) < 32:
            return 0.0
        
        # Create dataset and dataloader
        dataset = SelfPlayDataset(self.data_buffer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for states, policies, values in dataloader:
                # Forward pass
                policy_logits, value_pred = self.model(states)
                
                # Policy loss (cross-entropy)
                policy_loss = F.cross_entropy(policy_logits, policies.argmax(dim=1))
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred.squeeze(), values)
                
                # Combined loss
                loss = policy_loss + 1.0 * value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(dataloader)
        
        self.scheduler.step()
        
        avg_loss = total_loss / max(epochs, 1)
        
        if verbose:
            print(f"Training: {len(self.data_buffer)} samples, Loss: {avg_loss:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return avg_loss
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'games_collected': self.game_count,
            'total_training_games': self.total_games,
            'black_wins': self.win_count[BLACK_STONE],
            'white_wins': self.win_count[WHITE_STONE],
            'buffer_size': len(self.data_buffer),
            'win_rate_black': self.win_count[BLACK_STONE] / max(self.game_count, 1) * 100,
        }

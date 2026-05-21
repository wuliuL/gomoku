#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gomoku Game System - with Deep Learning AI
Supports PVP, PVE (3 difficulties), Deep Learning AI
"""

import pygame
import sys
import threading
import time
import math
import os

# Try to import deep learning dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

pygame.init()

# ============== Constants ==============
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700
INFO_HEIGHT = 60
WINDOW_SIZE = WINDOW_WIDTH

BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 40

# Colors
BG_COLOR = (210, 180, 140)
BOARD_BG = (230, 200, 160)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
INFO_BG = (60, 60, 60)
BUTTON_BG = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)
BUTTON_DL = (138, 43, 226)
BUTTON_DL_HOVER = (170, 70, 255)
BUTTON_TRAIN = (46, 139, 87)
BUTTON_TRAIN_HOVER = (60, 160, 100)
TEXT_COLOR = (240, 240, 240)
HINT_COLOR = (255, 215, 0)
WIN_COLOR = (255, 69, 0)

# Game constants
EMPTY = 0
BLACK_STONE = 1
WHITE_STONE = 2

# AI difficulty
DIFFICULTY_EASY = 1
DIFFICULTY_MEDIUM = 2
DIFFICULTY_HARD = 3
DIFFICULTY_DL = 4

# Evaluation function scores
SCORE_FIVE = 100000
SCORE_ALIVE_FOUR = 10000
SCORE_SLEEP_FOUR = 5000
SCORE_ALIVE_THREE = 5000
SCORE_SLEEP_THREE = 200
SCORE_ALIVE_TWO = 100
SCORE_SLEEP_TWO = 10

# ============== Deep Learning Module ==============
# 导入训练模块中的网络定义
import dl_model as dl_module
ResidualBlock = dl_module.ResidualBlock
GomokuNet = dl_module.GomokuNet


# ============== Helper Functions ==============
def screen_to_board(x, y):
    """Screen coordinates -> Board coordinates"""
    board_y = y - INFO_HEIGHT
    i = round((board_y - MARGIN) / CELL_SIZE)
    j = round((x - MARGIN) / CELL_SIZE)
    if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
        return i, j
    return None, None


def board_to_screen(i, j):
    """Board coordinates -> Screen coordinates"""
    x = MARGIN + j * CELL_SIZE
    y = INFO_HEIGHT + MARGIN + i * CELL_SIZE
    return x, y


# ============== AI Evaluator ==============
class GomokuEvaluator:
    """Gomoku pattern evaluator"""
    
    @staticmethod
    def evaluate_line(count, empty_ends):
        """Evaluate a line pattern"""
        if count >= 5:
            return SCORE_FIVE
        elif count == 4:
            if empty_ends == 2:
                return SCORE_ALIVE_FOUR
            elif empty_ends == 1:
                return SCORE_SLEEP_FOUR
        elif count == 3:
            if empty_ends == 2:
                return SCORE_ALIVE_THREE
            elif empty_ends == 1:
                return SCORE_SLEEP_THREE
        elif count == 2:
            if empty_ends == 2:
                return SCORE_ALIVE_TWO
            elif empty_ends == 1:
                return SCORE_SLEEP_TWO
        return 0
    
    @staticmethod
    def evaluate_board(board, player):
        """Evaluate board score for specified player"""
        enemy = 3 - player
        total_score = 0
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] != EMPTY:
                    continue
                
                for dx, dy in directions:
                    line = []
                    for k in range(-4, 5):
                        ni, nj = i + dx * k, j + dy * k
                        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                            line.append(board[ni][nj])
                        else:
                            line.append(-1)
                    
                    count = 0
                    empty_ends = 0
                    
                    for k in range(4, -1, -1):
                        if line[k] == player:
                            count += 1
                        elif line[k] == EMPTY:
                            break
                    
                    if line[5] == EMPTY:
                        empty_ends += 1
                    if line[-1] == EMPTY:
                        empty_ends += 1
                    
                    score = GomokuEvaluator.evaluate_line(count, empty_ends)
                    
                    if board[i][j] == EMPTY:
                        total_score += score * 0.1
        
        return total_score


# ============== Simple AI (Minimax) ==============
class SimpleAI:
    """Simple AI based on evaluation function"""
    
    def __init__(self, depth):
        self.depth = depth
    
    def get_candidates(self, board):
        """Get candidate move positions"""
        candidates = set()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] != EMPTY:
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                                if board[ni][nj] == EMPTY:
                                    candidates.add((ni, nj))
        return list(candidates)
    
    def count_line(self, board, i, j, player, dx, dy):
        """Count consecutive stones in one direction"""
        count = 1
        empty_ends = 0
        
        x, y = i + dx, j + dy
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if board[x][y] == player:
                count += 1
            elif board[x][y] == EMPTY:
                empty_ends = 1
                break
            else:
                break
            x, y = x + dx, y + dy
        
        x, y = i - dx, j - dy
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if board[x][y] == player:
                count += 1
            elif board[x][y] == EMPTY:
                empty_ends += 1
                break
            else:
                break
            x, y = x - dx, y - dy
        
        return count, empty_ends
    
    def evaluate_position(self, board, i, j, player):
        """Evaluate position value for player"""
        total = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count, empty_ends = self.count_line(board, i, j, player, dx, dy)
            total += GomokuEvaluator.evaluate_line(count, empty_ends)
        
        return total
    
    def get_best_move(self, board, player):
        """Get best move position"""
        candidates = self.get_candidates(board)
        if not candidates:
            return 7, 7
        
        # Check winning move
        for i, j in candidates:
            board[i][j] = player
            if self.check_five(board, i, j, player):
                board[i][j] = EMPTY
                return i, j
            board[i][j] = EMPTY
        
        # Check defense move
        enemy = 3 - player
        for i, j in candidates:
            board[i][j] = enemy
            if self.check_five(board, i, j, enemy):
                board[i][j] = EMPTY
                return i, j
            board[i][j] = EMPTY
        
        # Use evaluation function
        best_score = -float('inf')
        best_move = candidates[0]
        
        for i, j in candidates[:15]:
            attack_score = self.evaluate_position(board, i, j, player)
            defend_score = self.evaluate_position(board, i, j, enemy)
            total_score = attack_score + defend_score * 0.9
            
            if total_score > best_score:
                best_score = total_score
                best_move = (i, j)
        
        return best_move
    
    @staticmethod
    def check_five(board, i, j, player):
        """Check if five in a row"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            x, y = i + dx, j + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == player:
                count += 1
                x, y = x + dx, y + dy
            x, y = i - dx, j - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == player:
                count += 1
                x, y = x - dx, y - dy
            if count >= 5:
                return True
        return False


# ============== Deep Learning AI ==============
class DeepLearningAI:
    """Deep Learning AI using Neural Network"""
    
    _instance = None  # Shared model
    
    def __init__(self):
        self.model = None
        self.device = None
        
        # Try to load shared model or create new one
        if DeepLearningAI._instance is not None:
            self.model = DeepLearningAI._instance
        else:
            self._init_model()
    
    def _init_model(self):
        """Initialize neural network model"""
        if HAS_TORCH:
            self.model = GomokuNet(channels=128, num_blocks=10)
            self.device = next(self.model.parameters()).device
            
            # Try to load saved checkpoint (prioritize checkpoint over model)
            checkpoint_paths = [
                'models/gomoku_checkpoint.pth',
                'models/gomoku_model.pth',
                'models/gomoku_net_final.pt',
            ]
            
            for path in checkpoint_paths:
                if os.path.exists(path):
                    try:
                        checkpoint = torch.load(path, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            self.model.load_state_dict(checkpoint)
                        self.model.eval()
                        print(f"Loaded DL model from {path}")
                        DeepLearningAI._instance = self.model
                        return
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")
            
            print("No DL model found, using untrained model")
            DeepLearningAI._instance = self.model
        else:
            print("PyTorch not available, DL AI unavailable")
    
    @classmethod
    def has_model(cls):
        """Check if model is available"""
        return HAS_TORCH and cls._instance is not None
    
    def get_candidates(self, board):
        """Get candidate positions"""
        candidates = set()
        has_stones = False
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] != EMPTY:
                    has_stones = True
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                                if board[ni][nj] == EMPTY:
                                    candidates.add((ni, nj))
        
        if not has_stones:
            return [(7, 7)]
        
        if not candidates:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if board[i][j] == EMPTY:
                        candidates.add((i, j))
        
        return list(candidates)
    
    def predict_move(self, board, player):
        """Get AI's best move using neural network"""
        if self.model is not None:
            return self._get_dl_move(board, player)
        else:
            return self._fallback_move(board, player)
    
    def _get_dl_move(self, board, player):
        """Get move using deep learning"""
        valid_moves = self.get_candidates(board)
        if not valid_moves:
            return 7, 7
        
        # Prepare state
        state = self._get_state(board, player)
        return self.model.get_action(state, valid_moves)
    
    def _get_state(self, board, player):
        """Get neural network input state"""
        state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        opponent = WHITE_STONE if player == BLACK_STONE else BLACK_STONE
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == player:
                    state[0, i, j] = 1.0
                elif board[i][j] == opponent:
                    state[1, i, j] = 1.0
        
        state[2] = 1.0 if player == BLACK_STONE else 0.0
        return state
    
    def _fallback_move(self, board, player):
        """Fallback move when neural network is not available"""
        candidates = self.get_candidates(board)
        if not candidates:
            return 7, 7
        
        # Check winning move
        for i, j in candidates:
            board[i][j] = player
            if SimpleAI.check_five(board, i, j, player):
                board[i][j] = EMPTY
                return i, j
            board[i][j] = EMPTY
        
        # Check defense move
        enemy = 3 - player
        for i, j in candidates:
            board[i][j] = enemy
            if SimpleAI.check_five(board, i, j, enemy):
                board[i][j] = EMPTY
                return i, j
            board[i][j] = EMPTY
        
        # Random move
        import random
        return random.choice(candidates)


# ============== Training Interface ==============
class TrainingInterface:
    """Training interface for Deep Learning AI with cumulative training support"""
    
    def __init__(self, game):
        self.game = game
        self.visible = False
        self.is_training = False
        self.trainer = None
        self.training_thread = None
        
        # Training stats
        self.epoch = 0
        self.total_epochs = 100
        self.games_played = 0
        self.loss = 0.0
        self.win_rate = 0.0
        self.total_games = 0  # 累积训练总场次
        self.buffer_size = 0  # 数据缓冲区大小
        self.logs = []
        
        # Buttons
        self.buttons = {}
        self._init_buttons()
    
    def _init_buttons(self):
        """Initialize training interface buttons"""
        btn_w, btn_h = 100, 40
        
        self.buttons = {
            'start': pygame.Rect(WINDOW_WIDTH // 2 - btn_w * 2 - 20, WINDOW_HEIGHT - 100, btn_w, btn_h),
            'stop': pygame.Rect(WINDOW_WIDTH // 2 - btn_w // 2, WINDOW_HEIGHT - 100, btn_w, btn_h),
            'back': pygame.Rect(WINDOW_WIDTH // 2 + btn_w + 20, WINDOW_HEIGHT - 100, btn_w, btn_h),
            'fight': pygame.Rect(WINDOW_WIDTH - 120, WINDOW_HEIGHT - 60, 100, 40),
        }
    
    def show(self):
        """Show training interface"""
        self.visible = True
        self._init_buttons()
    
    def hide(self):
        """Hide training interface"""
        self.visible = False
    
    def handle_click(self, pos):
        """Handle click events"""
        x, y = pos
        
        if self.buttons['start'].collidepoint(x, y):
            if not self.is_training:
                self._start_training()
            return True
        
        if self.buttons['stop'].collidepoint(x, y):
            if self.is_training:
                self._stop_training()
            return True
        
        if self.buttons['back'].collidepoint(x, y):
            if self.is_training:
                self._stop_training()
            self.hide()
            self.game.state = 'menu'
            return True
        
        if self.buttons['fight'].collidepoint(x, y):
            if self.trainer and self.trainer.model:
                DeepLearningAI._instance = self.trainer.model
            self.hide()
            self.game.state = 'playing'
            self.game.mode = 'dl'
            self.game.ai = DeepLearningAI()
            self.game.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            self.game.current_player = BLACK_STONE
            self.game.game_over = False
            self.game.winner = None
            self.game.move_history = []
            self.game.last_move = None
            return True
        
        return False
    
    def _start_training(self):
        """Start training in background thread"""
        if self.is_training:
            return
        
        if not HAS_TORCH:
            self.logs.append("Error: PyTorch not available")
            return
        
        self.is_training = True
        
        # Initialize trainer (will load existing checkpoint if available)
        if self.trainer is None:
            from dl_model import SelfPlayTrainer
            self.trainer = SelfPlayTrainer(save_dir='models')
            
            # Load previous training stats
            stats = self.trainer.get_stats()
            self.total_games = stats['total_training_games']
            if self.total_games > 0:
                self.logs.append(f"Loaded previous training: {self.total_games} games collected")
        
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def _training_loop(self):
        """Training loop (runs in background) with cumulative training support"""
        for epoch in range(self.total_epochs):
            if not self.is_training:
                break
            
            # Collect self-play games
            self.trainer.collect_games(3, verbose=False)
            self.games_played = self.trainer.game_count
            
            # Train
            loss = self.trainer.train(epochs=1, batch_size=64)
            self.loss = loss
            self.epoch = epoch + 1
            
            # Save checkpoint periodically (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                self.trainer.save_checkpoint()
            
            # Update stats
            stats = self.trainer.get_stats()
            self.total_games = stats['total_training_games']
            self.buffer_size = stats['buffer_size']
            if stats['games_collected'] > 0:
                self.win_rate = stats['win_rate_black']
            
            # Log
            log = f"Epoch {epoch + 1}: Loss={loss:.4f}, Buffer={self.buffer_size}, TotalGames={self.total_games}"
            self.logs.append(log)
            if len(self.logs) > 20:
                self.logs.pop(0)
        
        # Save checkpoint on completion
        if self.trainer:
            self.trainer.save_checkpoint()
            self.trainer.save_model()
            DeepLearningAI._instance = self.trainer.model
        
        self.is_training = False
        self.logs.append("Training completed!")
    
    def _stop_training(self):
        """Stop training and save checkpoint"""
        self.is_training = False
        
        if self.trainer:
            self.trainer.save_checkpoint()  # Save checkpoint with stats
            self.trainer.save_model()
            DeepLearningAI._instance = self.trainer.model
        
        self.logs.append(f"Training stopped. Total games: {self.total_games}")
    
    def draw(self, screen):
        """Draw training interface"""
        # Background
        screen.fill((40, 44, 52))
        
        # Title
        font_large = pygame.font.Font(None, 56)
        font = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 22)
        
        title = font_large.render('Deep Learning Training', True, (255, 255, 255))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        # Status
        status = 'Training...' if self.is_training else 'Idle'
        status_color = (100, 255, 100) if self.is_training else (200, 200, 200)
        text = font.render(f'Status: {status}', True, status_color)
        screen.blit(text, (50, 110))
        
        # Progress
        progress = self.epoch / self.total_epochs
        bar_width, bar_height = 400, 25
        bar_x, bar_y = 50, 150
        
        pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=5)
        
        pct_text = font.render(f'{progress * 100:.1f}%', True, (255, 255, 255))
        screen.blit(pct_text, (bar_x + bar_width + 20, bar_y))
        
        # Stats
        y = 200
        stats = [
            f"Epoch: {self.epoch} / {self.total_epochs}",
            f"Session Games: {self.games_played}",
            f"Total Training Games: {self.total_games}",
            f"Data Buffer: {self.buffer_size}",
            f"Current Loss: {self.loss:.4f}" if self.loss > 0 else "Loss: N/A",
            f"Black Win Rate: {self.win_rate:.1f}%",
        ]
        
        for line in stats:
            text = font.render(line, True, (220, 220, 220))
            screen.blit(text, (50, y))
            y += 35
        
        # Logs
        log_y = 350
        log_title = font.render('Training Logs:', True, (180, 180, 180))
        screen.blit(log_title, (50, log_y))
        log_y += 30
        
        for log in self.logs[-12:]:
            log_text = font_small.render(log, True, (150, 150, 150))
            screen.blit(log_text, (50, log_y))
            log_y += 22
        
        # DL Status
        dl_status = "PyTorch Ready" if HAS_TORCH else "PyTorch Not Available"
        dl_color = (100, 255, 100) if HAS_TORCH else (255, 100, 100)
        dl_text = font_small.render(f'DL Status: {dl_status}', True, dl_color)
        screen.blit(dl_text, (50, WINDOW_HEIGHT - 30))
        
        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        
        # Start/Stop button
        btn = self.buttons['start']
        if not self.is_training:
            color = BUTTON_HOVER if btn.collidepoint(mouse_pos) else BUTTON_BG
            pygame.draw.rect(screen, color, btn, border_radius=8)
            text = font.render('Start', True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=btn.center))
        
        btn = self.buttons['stop']
        color = (180, 80, 80) if self.is_training else (80, 80, 80)
        if self.is_training and btn.collidepoint(mouse_pos):
            color = (200, 100, 100)
        pygame.draw.rect(screen, color, btn, border_radius=8)
        text = font.render('Stop', True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=btn.center))
        
        # Back button
        btn = self.buttons['back']
        color = BUTTON_HOVER if btn.collidepoint(mouse_pos) else BUTTON_BG
        pygame.draw.rect(screen, color, btn, border_radius=8)
        text = font.render('Back', True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=btn.center))
        
        # Fight AI button
        btn = self.buttons['fight']
        color = BUTTON_DL_HOVER if btn.collidepoint(mouse_pos) else BUTTON_DL
        pygame.draw.rect(screen, color, btn, border_radius=8)
        text = font.render('Fight AI', True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=btn.center))


# ============== Game Main Class ==============
class GomokuGame:
    """Gomoku Game Main Class"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Gomoku Game - Deep Learning AI')
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        self.font_title = pygame.font.Font(None, 56)
        
        # Game state
        self.state = 'menu'  # menu, playing, game_over, training
        self.mode = None  # pvp, pve, dl
        self.difficulty = DIFFICULTY_MEDIUM
        self.ai = None
        
        # Board data
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = BLACK_STONE
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.last_move = None
        
        # UI elements
        self.buttons = []
        self.hover_button = None
        self.difficulty_buttons = []
        
        # Training interface
        self.training = TrainingInterface(self)
        
        # Initialize menu
        self._init_menu()
    
    def _init_menu(self):
        """Initialize menu buttons"""
        self.buttons = []
        
        button_width = 280
        button_height = 55
        start_x = (WINDOW_WIDTH - button_width) // 2
        start_y = 200
        spacing = 75
        
        # PVP mode button
        self.buttons.append({
            'rect': pygame.Rect(start_x, start_y, button_width, button_height),
            'text': 'Player vs Player (PVP)',
            'action': 'pvp'
        })
        
        # PVE mode button
        self.buttons.append({
            'rect': pygame.Rect(start_x, start_y + spacing, button_width, button_height),
            'text': 'Player vs AI (PVE)',
            'action': 'pve_select'
        })
        
        # Deep Learning button
        self.buttons.append({
            'rect': pygame.Rect(start_x, start_y + spacing * 2, button_width, button_height),
            'text': 'Deep Learning AI',
            'action': 'dl',
            'is_dl': True
        })
        
        # Train AI button
        self.buttons.append({
            'rect': pygame.Rect(start_x, start_y + spacing * 3, button_width, button_height),
            'text': 'Train AI Model',
            'action': 'train',
            'is_train': True
        })
        
        # Difficulty selection buttons
        self._init_difficulty_select()
    
    def _init_difficulty_select(self):
        """Initialize difficulty selection buttons"""
        self.difficulty_buttons = []
        
        button_width = 100
        button_height = 40
        center_x = (WINDOW_WIDTH - button_width * 4 - 30) // 2
        start_y = 310
        spacing = 110
        
        difficulties = [
            ('Easy', DIFFICULTY_EASY),
            ('Medium', DIFFICULTY_MEDIUM),
            ('Hard', DIFFICULTY_HARD),
        ]
        
        for i, (text, action) in enumerate(difficulties):
            self.difficulty_buttons.append({
                'rect': pygame.Rect(center_x + i * spacing, start_y, button_width, button_height),
                'text': text,
                'action': action
            })
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_hover(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)
            
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()
    
    def _handle_click(self, pos):
        """Handle mouse click"""
        x, y = pos
        
        if self.state == 'training':
            if self.training.handle_click(pos):
                return
            return
        
        if self.state == 'menu':
            for btn in self.buttons:
                if btn['rect'].collidepoint(pos):
                    self._execute_menu_action(btn['action'])
                    return
            
            # Difficulty selection
            for btn in self.difficulty_buttons:
                if btn['rect'].collidepoint(pos):
                    self.difficulty = btn['action']
                    return
        
        elif self.state == 'playing':
            if y > INFO_HEIGHT:
                i, j = screen_to_board(x, y)
                if i is not None and not self.game_over:
                    if self.board[i][j] == EMPTY:
                        self._place_stone(i, j)
            
            self._handle_game_button_click(pos)
        
        elif self.state == 'game_over':
            self._return_to_menu()
    
    def _handle_hover(self, pos):
        """Handle mouse hover"""
        if self.state == 'menu':
            self.hover_button = None
            for btn in self.buttons:
                if btn['rect'].collidepoint(pos):
                    self.hover_button = btn
                    break
    
    def _handle_key(self, event):
        """Handle keyboard events"""
        if event.key == pygame.K_ESCAPE:
            if self.state == 'playing' or self.state == 'game_over':
                self._return_to_menu()
        elif event.key == pygame.K_r:
            if self.state == 'playing':
                self._restart_game()
    
    def _execute_menu_action(self, action):
        """Execute menu action"""
        if action == 'pvp':
            self._start_game('pvp')
        elif action == 'pve_select':
            self._start_game('pve')
        elif action == 'dl':
            self._start_game('dl')
        elif action == 'train':
            self.training.show()
            self.state = 'training'
    
    def _start_game(self, mode):
        """Start game"""
        self.mode = mode
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = BLACK_STONE
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.last_move = None
        self.state = 'playing'
        
        if mode == 'pve':
            self.ai = SimpleAI(self.difficulty)
        elif mode == 'dl':
            self.ai = DeepLearningAI()
        else:
            self.ai = None
    
    def _place_stone(self, i, j):
        """Place stone"""
        self.board[i][j] = self.current_player
        self.move_history.append((i, j, self.current_player))
        self.last_move = (i, j)
        
        if self._check_win(i, j, self.current_player):
            self.game_over = True
            self.winner = self.current_player
            self.state = 'game_over'
            return
        
        self.current_player = 3 - self.current_player
        
        if not self.game_over and (self.mode == 'pve' or self.mode == 'dl'):
            self._ai_move()
    
    def _ai_move(self):
        """AI move"""
        if self.mode == 'pve':
            move = self.ai.get_best_move(self.board, self.current_player)
        elif self.mode == 'dl':
            move = self.ai.predict_move(self.board, self.current_player)
        else:
            return
        
        i, j = move
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            if self.board[i][j] == EMPTY:
                self.board[i][j] = self.current_player
                self.move_history.append((i, j, self.current_player))
                self.last_move = (i, j)
                
                if self._check_win(i, j, self.current_player):
                    self.game_over = True
                    self.winner = self.current_player
                    self.state = 'game_over'
                    return
                
                self.current_player = 3 - self.current_player
    
    def _check_win(self, i, j, player):
        """Check win"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            x, y = i + dx, j + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x, y = x + dx, y + dy
            x, y = i - dx, j - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x, y = x - dx, y - dy
            if count >= 5:
                return True
        return False
    
    def _handle_game_button_click(self, pos):
        """Handle game button click"""
        x, y = pos
        button_y = WINDOW_HEIGHT - 50
        button_width = 100
        button_height = 40
        
        if WINDOW_WIDTH - button_width - 20 <= x <= WINDOW_WIDTH - 20 and button_y <= y <= button_y + button_height:
            self._return_to_menu()
        
        if WINDOW_WIDTH - button_width * 2 - 40 <= x <= WINDOW_WIDTH - button_width * 2 - 20 and button_y <= y <= button_y + button_height:
            self._restart_game()
    
    def _restart_game(self):
        """Restart game"""
        self._start_game(self.mode)
    
    def _return_to_menu(self):
        """Return to menu"""
        self.state = 'menu'
        self.mode = None
        self.difficulty = DIFFICULTY_MEDIUM
        self.ai = None
    
    def _draw(self):
        """Draw interface"""
        if self.state == 'training':
            self.training.draw(self.screen)
        elif self.state == 'menu':
            self._draw_menu()
        elif self.state == 'playing' or self.state == 'game_over':
            self._draw_game()
    
    def _draw_menu(self):
        """Draw menu interface"""
        self.screen.fill(BG_COLOR)
        
        # Title
        title = self.font_title.render('Gomoku Game', True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = self.font.render('Human vs AI Battle System', True, (100, 100, 100))
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 120))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Draw buttons
        for btn in self.buttons:
            is_hover = self.hover_button == btn
            
            if btn.get('is_dl'):
                color = BUTTON_DL_HOVER if is_hover else BUTTON_DL
            elif btn.get('is_train'):
                color = BUTTON_TRAIN_HOVER if is_hover else BUTTON_TRAIN
            else:
                color = BUTTON_HOVER if is_hover else BUTTON_BG
            
            pygame.draw.rect(self.screen, color, btn['rect'], border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), btn['rect'], 2, border_radius=10)
            
            text = self.font.render(btn['text'], True, WHITE)
            text_rect = text.get_rect(center=btn['rect'].center)
            self.screen.blit(text, text_rect)
        
        # Difficulty selection hint
        hint = self.font.render('Select difficulty below:', True, (120, 120, 120))
        hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, 290))
        self.screen.blit(hint, hint_rect)
        
        # Difficulty buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.difficulty_buttons:
            is_selected = self.difficulty == btn['action']
            is_hover = btn['rect'].collidepoint(mouse_pos)
            
            if is_selected:
                color = (100, 130, 100)
            elif is_hover:
                color = BUTTON_HOVER
            else:
                color = BUTTON_BG
            
            pygame.draw.rect(self.screen, color, btn['rect'], border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255), btn['rect'], 1, border_radius=8)
            
            text = self.font.render(btn['text'], True, WHITE)
            text_rect = text.get_rect(center=btn['rect'].center)
            self.screen.blit(text, text_rect)
        
        # Bottom hint
        hint = self.font.render('Click to start | ESC to return', True, (120, 120, 120))
        hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
        self.screen.blit(hint, hint_rect)
    
    def _draw_game(self):
        """Draw game interface"""
        self.screen.fill(BOARD_BG)
        
        self._draw_info_bar()
        self._draw_board()
        self._draw_stones()
        
        if self.last_move:
            self._draw_last_move_marker(self.last_move[0], self.last_move[1])
        
        if self.game_over:
            self._draw_win_message()
        
        self._draw_game_buttons()
    
    def _draw_info_bar(self):
        """Draw info bar"""
        pygame.draw.rect(self.screen, INFO_BG, (0, 0, WINDOW_WIDTH, INFO_HEIGHT))
        
        player_text = 'Black' if self.current_player == BLACK_STONE else 'White'
        mode_text = ''
        
        if self.mode == 'pvp':
            mode_text = 'Player vs Player'
        elif self.mode == 'pve':
            diff_names = {DIFFICULTY_EASY: 'Easy', DIFFICULTY_MEDIUM: 'Medium', DIFFICULTY_HARD: 'Hard'}
            mode_text = f'Player vs AI - {diff_names.get(self.difficulty, "Easy")}'
        elif self.mode == 'dl':
            mode_text = 'Deep Learning AI'
        
        info = f'{mode_text} | Turn: {player_text}'
        text = self.font.render(info, True, TEXT_COLOR)
        self.screen.blit(text, (20, 20))
        
        steps = len(self.move_history)
        steps_text = self.font.render(f'Moves: {steps}', True, TEXT_COLOR)
        steps_rect = steps_text.get_rect(center=(WINDOW_WIDTH - 80, INFO_HEIGHT // 2))
        self.screen.blit(steps_text, steps_rect)
    
    def _draw_board(self):
        """Draw board"""
        for i in range(BOARD_SIZE):
            start_x = MARGIN
            end_x = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
            y = MARGIN + i * CELL_SIZE
            pygame.draw.line(self.screen, BLACK, (start_x, y), (end_x, y), 1)
            
            start_y = MARGIN
            end_y = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
            x = MARGIN + i * CELL_SIZE
            pygame.draw.line(self.screen, BLACK, (x, start_y), (x, end_y), 1)
        
        star_positions = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for i, j in star_positions:
            x, y = board_to_screen(i, j)
            pygame.draw.circle(self.screen, BLACK, (x, y), 5)
    
    def _draw_stones(self):
        """Draw stones"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] != EMPTY:
                    x, y = board_to_screen(i, j)
                    color = BLACK if self.board[i][j] == BLACK_STONE else WHITE
                    
                    pygame.draw.circle(self.screen, (150, 150, 150), (x + 2, y + 2), 15)
                    pygame.draw.circle(self.screen, color, (x, y), 15)
                    pygame.draw.circle(self.screen, BLACK if color == WHITE else (80, 80, 80), (x, y), 15, 1)
    
    def _draw_last_move_marker(self, i, j):
        """Draw last move marker"""
        x, y = board_to_screen(i, j)
        pygame.draw.circle(self.screen, HINT_COLOR, (x, y), 18, 3)
    
    def _draw_win_message(self):
        """Draw win message"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        if self.winner == BLACK_STONE:
            winner_text = 'Black Wins!'
        else:
            winner_text = 'White Wins!'
        
        text = self.font_large.render(winner_text, True, WIN_COLOR)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(text, text_rect)
        
        hint = self.font.render('Click anywhere to return to menu', True, WHITE)
        hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
        self.screen.blit(hint, hint_rect)
    
    def _draw_game_buttons(self):
        """Draw game buttons"""
        button_y = WINDOW_HEIGHT - 50
        button_width = 100
        button_height = 40
        
        menu_rect = pygame.Rect(WINDOW_WIDTH - button_width - 20, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, (100, 100, 100), menu_rect, border_radius=8)
        text = self.font.render('Menu', True, WHITE)
        text_rect = text.get_rect(center=menu_rect.center)
        self.screen.blit(text, text_rect)
        
        restart_rect = pygame.Rect(WINDOW_WIDTH - button_width * 2 - 40, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, (100, 100, 100), restart_rect, border_radius=8)
        text = self.font.render('Restart', True, WHITE)
        text_rect = text.get_rect(center=restart_rect.center)
        self.screen.blit(text, text_rect)


# ============== Entry Point ==============
if __name__ == '__main__':
    game = GomokuGame()
    game.run()

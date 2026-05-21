import pygame
import sys
import threading
import io
import os

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# 尝试加载深度学习模块
HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    HAS_TORCH = True
except ImportError:
    pass

pygame.init()

# ================== 音效 ==================
def generate_sound(frequency, duration, volume=0.5):
    """生成简单的音效"""
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    
    if HAS_NUMPY:
        t = np.linspace(0, duration, n_samples, False)
        # 生成正弦波并添加衰减
        wave = np.sin(2 * np.pi * frequency * t)
        wave = wave * np.exp(-t / (duration * 0.3))  # 衰减
        wave = (wave * volume * 32767).astype(np.int16)
        # 转换为立体声
        stereo_wave = np.column_stack((wave, wave))
        sound_data = stereo_wave.tobytes()
        sound_surface = pygame.Surface((len(sound_data), 1))
        return pygame.mixer.Sound(buffer=sound_data)
    else:
        # 无numpy时的简单替代方案（使用pygame内置）
        pass

# 尝试初始化音效系统
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    sound_enabled = True
    # 生成落子音效
    stone_sound = generate_sound(800, 0.1, 0.3)
    win_sound = generate_sound(523, 0.3, 0.5)  # C5音符
except:
    sound_enabled = False
    stone_sound = None
    win_sound = None

def play_stone_sound():
    """播放落子音效"""
    if sound_enabled and stone_sound:
        stone_sound.play()

def play_win_sound():
    """播放胜利音效"""
    if sound_enabled and win_sound:
        win_sound.play()

# ================== 常量 ==================
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 40
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
INFO_HEIGHT = 70
WINDOW_HEIGHT = WINDOW_SIZE + INFO_HEIGHT

# 颜色 - 现代配色
BOARD_BG = (222, 184, 135)
LINE_COLOR = (80, 60, 40)
BLACK = (20, 20, 20)
WHITE = (250, 250, 250)
INFO_BG = (45, 45, 55)
INFO_TEXT = (220, 220, 220)
BUTTON_COLOR = (70, 70, 85)
BUTTON_HOVER = (90, 90, 110)
BUTTON_ACTIVE = (100, 130, 100)
MENU_BG = (30, 30, 40)
ACCENT_COLOR = (255, 180, 80)
WIN_COLOR = (255, 100, 100)

EMPTY = 0
BLACK_STONE = 1
WHITE_STONE = 2

# 棋型分值
SCORES = {
    'FIVE': 100000,
    'ALIVE_FOUR': 10000,
    'SLEEP_FOUR': 5000,
    'ALIVE_THREE': 5000,
    'SLEEP_THREE': 200,
    'ALIVE_TWO': 100,
    'SLEEP_TWO': 10,
}

# 难度等级
DIFFICULTY_EASY = 1
DIFFICULTY_MEDIUM = 2
DIFFICULTY_HARD = 3
DIFFICULTY_DL = 4  # 深度学习AI

# AI类型
AI_TYPE_MINIMAX = 'minimax'
AI_TYPE_DL = 'deep_learning'

# 难度对应的搜索深度
DIFFICULTY_DEPTHS = {
    DIFFICULTY_EASY: 1,
    DIFFICULTY_MEDIUM: 2,
    DIFFICULTY_HARD: 3,
}

DIFFICULTY_NAMES = {
    DIFFICULTY_EASY: "Easy",
    DIFFICULTY_MEDIUM: "Medium",
    DIFFICULTY_HARD: "Hard",
    DIFFICULTY_DL: "Deep AI",
}

# ============ 深度学习网络定义 ============
if HAS_TORCH:
    class ResBlock(torch.nn.Module):
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
        def __init__(self, board_size=15, channels=128, num_blocks=10):
            super().__init__()
            self.board_size = board_size
            self.input_conv = torch.nn.Sequential(
                torch.nn.Conv2d(4, channels, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU()
            )
            self.res_tower = torch.nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
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
            self.eval()
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if len(state.shape) == 3:
                    state = state.unsqueeze(0)
                state = state.to(next(self.parameters()).device)
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


# ================== 自我对弈训练器 ==================
class SelfPlayTrainer:
    """自我对弈训练器"""
    
    _instance = None  # 共享模型
    
    def __init__(self, save_dir='models'):
        self.model = GomokuNet(channels=128, num_blocks=10)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.save_dir = save_dir
        self.game_count = 0
        self.win_count = {BLACK_STONE: 0, WHITE_STONE: 0}
        self.device = next(self.model.parameters()).device
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 尝试加载已有模型
        model_paths = [
            os.path.join(save_dir, 'gomoku_model.pth'),
            os.path.join(save_dir, 'gomoku_net_final.pt'),
        ]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location='cpu')
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        self.model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        self.model.load_state_dict(state_dict)
                    print(f"已加载模型: {path}")
                    break
                except Exception as e:
                    print(f"加载失败 {path}: {e}")
    
    def self_play_game(self, verbose=False):
        """自我对弈一局"""
        board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        current_player = BLACK_STONE
        move_history = []
        states = []
        policies = []
        winners = []
        
        for step in range(BOARD_SIZE * BOARD_SIZE):
            valid_moves = self._get_valid_moves(board)
            if not valid_moves:
                break
            
            # 获取网络落子
            move = self._get_network_move(board, current_player, valid_moves)
            if move is None:
                break
            
            i, j = move
            board[i][j] = current_player
            move_history.append((i, j, current_player))
            
            # 检查胜利
            if self._check_win(board, i, j, current_player):
                winner = current_player
                if verbose:
                    print(f"第{self.game_count}局: 玩家{current_player}获胜")
                break
            
            current_player = 3 - current_player
        else:
            winner = EMPTY
        
        # 记录状态和结果
        for idx, (i, j, player) in enumerate(move_history):
            state = self._get_state(board, player, idx, move_history)
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            policy[i * BOARD_SIZE + j] = 1.0
            
            reward = 1.0 if player == winner else (-1.0 if winner != EMPTY else 0.0)
            
            states.append(state)
            policies.append(policy)
            winners.append(reward)
        
        self.game_count += 1
        if winner != EMPTY:
            self.win_count[winner] += 1
        
        return states, policies, winners
    
    def _get_network_move(self, board, player, valid_moves):
        """从神经网络获取落子"""
        if not valid_moves:
            return None
        
        state = self._get_state(board, player, 0, [])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, _ = self.model(state_tensor)
            policy = F.softmax(policy_logits, dim=-1)[0].cpu().numpy()
            
            # 应用掩码
            mask = np.zeros_like(policy)
            for r, c in valid_moves:
                mask[r * BOARD_SIZE + c] = 1.0
            policy = policy * mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = mask / len(valid_moves)
            
            idx = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=policy)
            return (idx // BOARD_SIZE, idx % BOARD_SIZE)
    
    def _get_state(self, board, player, move_idx, move_history):
        """获取神经网络输入状态"""
        state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        opponent = WHITE_STONE if player == BLACK_STONE else BLACK_STONE
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == player:
                    state[0, i, j] = 1.0
                elif board[i][j] == opponent:
                    state[1, i, j] = 1.0
        
        state[2] = 1.0 if player == BLACK_STONE else 0.0
        
        if move_history and move_idx > 0:
            li, lj, _ = move_history[move_idx - 1]
            state[3, li, lj] = 1.0
        
        return state
    
    def _get_valid_moves(self, board):
        """获取有效落子位置"""
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
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        
        return list(candidates) if candidates else [(7, 7)]
    
    def _check_win(self, board, i, j, player):
        """检查是否获胜"""
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
    
    def collect_games(self, num_games=10, verbose=False):
        """收集自我对弈数据"""
        for _ in range(num_games):
            self.self_play_game(verbose=verbose)
    
    def train(self, states, policies, winners, epochs=1, batch_size=64):
        """训练模型"""
        if not states:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(states))
            
            for i in range(0, len(states), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_states = torch.FloatTensor(np.array([states[j] for j in batch_idx]))
                batch_policies = torch.FloatTensor(np.array([policies[j] for j in batch_idx]))
                batch_values = torch.FloatTensor(np.array([winners[j] for j in batch_idx]))
                
                self.optimizer.zero_grad()
                policy_logits, value_pred = self.model(batch_states)
                
                # 策略损失
                policy_loss = F.cross_entropy(policy_logits, batch_policies.argmax(dim=1))
                
                # 价值损失
                value_loss = F.mse_loss(value_pred.squeeze(), batch_values)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.eval()
        return total_loss / max(1, num_batches)
    
    def save_model(self, name='gomoku_model.pth'):
        """保存模型"""
        path = os.path.join(self.save_dir, name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'game_count': self.game_count,
            'win_count': self.win_count
        }, path)
        print(f"模型已保存: {path}")
        return path


# ================== 训练界面 ==================
TRAINING_WIDTH = 680
TRAINING_HEIGHT = 600
BUTTON_BG = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)
BUTTON_DL = (138, 43, 226)
BUTTON_DL_HOVER = (170, 70, 255)

# 添加缺失颜色常量到主模块
MENU_BG = (30, 30, 40)
ACCENT_COLOR = (255, 180, 80)
WIN_COLOR = (255, 100, 100)
BUTTON_COLOR = (70, 70, 85)
BUTTON_ACTIVE = (100, 130, 100)


class TrainingInterface:
    """训练界面"""
    
    def __init__(self, game):
        self.game = game
        self.visible = False
        self.is_training = False
        self.trainer = None
        self.training_thread = None
        self.collected_data = {'states': [], 'policies': [], 'winners': []}
        
        # 训练统计
        self.epoch = 0
        self.total_epochs = 100
        self.games_played = 0
        self.loss = 0.0
        self.win_rate = 0.0
        self.logs = []
        
        # 按钮
        self.buttons = {}
        self._init_buttons()
    
    def _init_buttons(self):
        """初始化按钮"""
        btn_w, btn_h = 100, 40
        
        self.buttons = {
            'start': pygame.Rect(TRAINING_WIDTH // 2 - btn_w * 2 - 20, TRAINING_HEIGHT - 100, btn_w, btn_h),
            'stop': pygame.Rect(TRAINING_WIDTH // 2 - btn_w // 2, TRAINING_HEIGHT - 100, btn_w, btn_h),
            'back': pygame.Rect(TRAINING_WIDTH // 2 + btn_w + 20, TRAINING_HEIGHT - 100, btn_w, btn_h),
            'fight': pygame.Rect(TRAINING_WIDTH - 120, TRAINING_HEIGHT - 60, 100, 40),
        }
    
    def show(self):
        """显示训练界面"""
        self.visible = True
        self._init_buttons()
        self.logs.append("=" * 30)
        self.logs.append("Deep Learning Training System")
        self.logs.append("=" * 30)
        if HAS_TORCH:
            self.logs.append("PyTorch: Ready")
        else:
            self.logs.append("PyTorch: Not Available")
    
    def hide(self):
        """隐藏训练界面"""
        self.visible = False
    
    def handle_click(self, pos):
        """处理点击事件"""
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
            return True
        
        if self.buttons['fight'].collidepoint(x, y):
            self._fight_ai()
            return True
        
        return False
    
    def _fight_ai(self):
        """开始与AI对战"""
        if self.trainer and self.trainer.model:
            SelfPlayTrainer._instance = self.trainer.model
            # 更新游戏类的深度学习网络
            self.game.dl_net = self.trainer.model
        
        self.hide()
        self.game.mode = 'pve'
        self.game.difficulty = DIFFICULTY_DL
        self.game.ai_type = AI_TYPE_DL
        self.game.return_to_game = True
    
    def _start_training(self):
        """开始训练"""
        if self.is_training:
            return
        
        if not HAS_TORCH:
            self.logs.append("Error: PyTorch not available")
            return
        
        self.is_training = True
        self.logs.append("Starting training...")
        
        # 初始化训练器
        self.trainer = SelfPlayTrainer(save_dir='models')
        self.collected_data = {'states': [], 'policies': [], 'winners': []}
        
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def _training_loop(self):
        """训练循环"""
        import random
        
        for epoch in range(self.total_epochs):
            if not self.is_training:
                break
            
            # 收集自我对弈数据
            for _ in range(5):
                states, policies, winners = self.trainer.self_play_game(verbose=False)
                self.collected_data['states'].extend(states)
                self.collected_data['policies'].extend(policies)
                self.collected_data['winners'].extend(winners)
                self.games_played = self.trainer.game_count
            
            # 训练
            if self.collected_data['states']:
                self.loss = self.trainer.train(
                    self.collected_data['states'],
                    self.collected_data['policies'],
                    self.collected_data['winners'],
                    epochs=1,
                    batch_size=64
                )
                
                # 限制数据量防止内存问题
                max_data = 50000
                if len(self.collected_data['states']) > max_data:
                    keep = len(self.collected_data['states']) - max_data
                    self.collected_data = {
                        'states': self.collected_data['states'][-keep:],
                        'policies': self.collected_data['policies'][-keep:],
                        'winners': self.collected_data['winners'][-keep:]
                    }
            
            self.epoch = epoch + 1
            
            # 更新胜率
            total = self.trainer.game_count
            if total > 0:
                self.win_rate = self.trainer.win_count[BLACK_STONE] / total * 100
            
            # Log entry
            log = f"Epoch {self.epoch}: Loss={self.loss:.4f}, Games={total}, Black Win Rate={self.win_rate:.1f}%"
            self.logs.append(log)
            if len(self.logs) > 20:
                self.logs.pop(0)
        
        # 保存最终模型
        if self.trainer:
            self.trainer.save_model()
            SelfPlayTrainer._instance = self.trainer.model
        
        self.is_training = False
        self.logs.append("Training completed!")
    
    def _stop_training(self):
        """停止训练"""
        self.is_training = False
        
        if self.trainer:
            self.trainer.save_model()
            SelfPlayTrainer._instance = self.trainer.model
        
        self.logs.append("Training stopped")
    
    def draw(self, screen):
        """绘制训练界面"""
        # 背景
        screen.fill((40, 44, 52))
        
        # 标题
        font_large = pygame.font.Font(None, 56)
        font = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 22)
        
        title = font_large.render('Deep Learning Training', True, (255, 255, 255))
        title_rect = title.get_rect(center=(TRAINING_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        # Status
        status = 'Training...' if self.is_training else 'Idle'
        status_color = (100, 255, 100) if self.is_training else (200, 200, 200)
        text = font.render(f'Status: {status}', True, status_color)
        screen.blit(text, (50, 110))
        
        # 进度条
        progress = self.epoch / self.total_epochs if self.total_epochs > 0 else 0
        bar_width, bar_height = 400, 25
        bar_x, bar_y = 50, 150
        
        pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=5)
        
        pct_text = font.render(f'{progress * 100:.1f}%', True, (255, 255, 255))
        screen.blit(pct_text, (bar_x + bar_width + 20, bar_y))
        
        # Statistics
        y = 200
        data_count = len(self.collected_data.get('states', []))
        stats = [
            f"Epoch: {self.epoch} / {self.total_epochs}",
            f"Games: {self.games_played}",
            f"Samples: {data_count}",
            f"Loss: {self.loss:.4f}" if self.loss > 0 else "Loss: N/A",
            f"Black Win Rate: {self.win_rate:.1f}%",
        ]
        
        for line in stats:
            text = font.render(line, True, (220, 220, 220))
            screen.blit(text, (50, y))
            y += 35
        
        # Log
        log_y = 380
        log_title = font.render('Training Log:', True, (180, 180, 180))
        screen.blit(log_title, (50, log_y))
        log_y += 30
        
        for log in self.logs[-10:]:
            log_text = font_small.render(log, True, (150, 150, 150))
            screen.blit(log_text, (50, log_y))
            log_y += 22
        
        # Deep learning status
        dl_status = "PyTorch Ready" if HAS_TORCH else "PyTorch Unavailable"
        dl_color = (100, 255, 100) if HAS_TORCH else (255, 100, 100)
        dl_text = font_small.render(f'DL Status: {dl_status}', True, dl_color)
        screen.blit(dl_text, (50, TRAINING_HEIGHT - 30))
        
        # 按钮
        mouse_pos = pygame.mouse.get_pos()
        
        # 开始按钮
        btn = self.buttons['start']
        if not self.is_training:
            color = BUTTON_HOVER if btn.collidepoint(mouse_pos) else BUTTON_BG
            pygame.draw.rect(screen, color, btn, border_radius=8)
            text = font.render('Start', True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=btn.center))
        
        # Stop button
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


class Gomoku:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_HEIGHT))
        pygame.display.set_caption("Gomoku - 五子棋 AI 训练系统")
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 22)
        self.tiny_font = pygame.font.Font(None, 18)
        
        # 完整离屏缓冲
        self.buffer = pygame.Surface((WINDOW_SIZE, WINDOW_HEIGHT))

        self.board = None
        self.current_player = None
        self.game_over = None
        self.winner = None
        self.mode = None
        self.difficulty = DIFFICULTY_MEDIUM
        self.return_to_menu = False
        self.return_to_game = False
        self.show_training = False
        self.ai_thinking = False
        self.move_history = []
        self.ai_result = None
        self.ai_lock = threading.Lock()
        self.board_needs_redraw = True
        self.info_needs_redraw = True
        self.sound_enabled = True
        self.last_move = None

        # 训练界面
        self.training_interface = TrainingInterface(self)
        self.training_screen = None

        # 菜单按钮
        self.btn_pvp = pygame.Rect(WINDOW_SIZE//2 - 130, WINDOW_SIZE//2 + 100, 120, 50)
        self.btn_pve = pygame.Rect(WINDOW_SIZE//2 + 10, WINDOW_SIZE//2 + 100, 120, 50)
        
        # 难度选择按钮（同一行排列）
        self.btn_easy = pygame.Rect(WINDOW_SIZE//2 - 220, WINDOW_SIZE//2 - 20, 100, 40)
        self.btn_medium = pygame.Rect(WINDOW_SIZE//2 - 110, WINDOW_SIZE//2 - 20, 100, 40)
        self.btn_hard = pygame.Rect(WINDOW_SIZE//2, WINDOW_SIZE//2 - 20, 100, 40)
        
        # 训练AI按钮（与难度按钮同一行）
        self.btn_train_ai = pygame.Rect(WINDOW_SIZE//2 + 110, WINDOW_SIZE//2 - 20, 110, 40)
        
        # 初始化深度学习网络
        self.dl_net = None
        self.dl_model_path = None
        self.ai_type = AI_TYPE_MINIMAX
        if HAS_TORCH:
            self._init_dl_network()

        # 游戏内按钮
        btn_w = 42
        btn_h = 32
        self.btn_menu = pygame.Rect(8, 20, btn_w, btn_h)
        self.btn_undo = pygame.Rect(56, 20, btn_w, btn_h)
        self.btn_sound = pygame.Rect(104, 20, btn_w, btn_h)
        self.btn_hint = pygame.Rect(152, 20, btn_w, btn_h)
        self.btn_restart = pygame.Rect(WINDOW_SIZE - 96, 20, btn_w, btn_h)
        self.btn_swap = pygame.Rect(WINDOW_SIZE - 48, 20, btn_w, btn_h)
        
        # 提示功能
        self.hint_enabled = False
        self.hint_position = None
        self.hint_thinking = False
    
    def _init_dl_network(self):
        """初始化深度学习网络"""
        # 多个可能的模型路径
        model_paths = [
            'models/gomoku_net_final.pt',
            '.venv/models/gomoku_net_final.pt',
            '../.venv/models/gomoku_net_final.pt',
            'models/gomoku_net_iter_10.pt',
            '.venv/models/gomoku_net_iter_10.pt',
        ]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.dl_net = GomokuNet(channels=128, num_blocks=10)
                    self.dl_net.load_state_dict(torch.load(path, map_location='cpu'))
                    self.dl_net.eval()
                    self.dl_model_path = path
                    print(f"Loaded DL model from {path}")
                    return
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        print("No DL model found, Deep AI unavailable")
    
    def get_state(self):
        """获取当前状态的神经网络输入格式"""
        state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        opponent = WHITE_STONE if self.current_player == BLACK_STONE else BLACK_STONE
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.current_player:
                    state[0, i, j] = 1.0
                elif self.board[i][j] == opponent:
                    state[1, i, j] = 1.0
        state[2] = 1.0 if self.current_player == BLACK_STONE else 0.0
        if self.move_history:
            li, lj, _ = self.move_history[-1]
            state[3, li, lj] = 1.0
        return state

    # ---------- 菜单 ----------
    def draw_menu(self):
        self.screen.fill(MENU_BG)
        
        title = self.font.render("Gomoku", True, ACCENT_COLOR)
        self.screen.blit(title, (WINDOW_SIZE//2 - title.get_width()//2, 60))
        subtitle = self.small_font.render("Five in a Row", True, (150,150,150))
        self.screen.blit(subtitle, (WINDOW_SIZE//2 - subtitle.get_width()//2, 95))
        
        difficulty_title = self.small_font.render("AI Difficulty", True, (180,180,180))
        self.screen.blit(difficulty_title, (WINDOW_SIZE//2 - difficulty_title.get_width()//2, WINDOW_SIZE//2 - 80))
        
        mouse_pos = pygame.mouse.get_pos()
        
        # 难度按钮列表（包含Train AI）
        diff_buttons = [
            (self.btn_easy, DIFFICULTY_EASY, "Easy"),
            (self.btn_medium, DIFFICULTY_MEDIUM, "Medium"),
            (self.btn_hard, DIFFICULTY_HARD, "Hard"),
        ]
        
        for btn, diff, name in diff_buttons:
            is_selected = self.difficulty == diff
            is_hover = btn.collidepoint(mouse_pos)
            if is_selected:
                color = BUTTON_ACTIVE
            elif is_hover:
                color = BUTTON_HOVER
            else:
                color = BUTTON_COLOR
            pygame.draw.rect(self.screen, color, btn, border_radius=8)
            pygame.draw.rect(self.screen, (255,255,255), btn, 1, 8)
            text = self.small_font.render(name, True, (255,255,255))
            self.screen.blit(text, (btn.x + (btn.width - text.get_width())//2, btn.y + 6))
        
        # Train AI按钮（与难度按钮同一行）
        if HAS_TORCH:
            is_hover = self.btn_train_ai.collidepoint(mouse_pos)
            color = BUTTON_HOVER if is_hover else BUTTON_DL
            pygame.draw.rect(self.screen, color, self.btn_train_ai, border_radius=8)
            pygame.draw.rect(self.screen, (255,255,255), self.btn_train_ai, 1, 8)
            text = self.small_font.render("Train AI", True, (255,255,255))
            self.screen.blit(text, (self.btn_train_ai.x + (self.btn_train_ai.width - text.get_width())//2, self.btn_train_ai.y + 6))
        
        text_mode = self.small_font.render("Game Mode", True, (180,180,180))
        self.screen.blit(text_mode, (WINDOW_SIZE//2 - text_mode.get_width()//2, WINDOW_SIZE//2 + 80))

        for btn, mode, name in [(self.btn_pvp, 'pvp', "PVP"), (self.btn_pve, 'pve', "PVE")]:
            is_hover = btn.collidepoint(mouse_pos)
            color = BUTTON_HOVER if is_hover else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, btn, border_radius=10)
            pygame.draw.rect(self.screen, (255,255,255), btn, 2, 10)
            text = self.font.render(name, True, (255,255,255))
            self.screen.blit(text, (btn.x + (btn.width - text.get_width())//2, btn.y + 13))

        pygame.display.flip()

    def run_menu(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # 难度选择
                    if self.btn_easy.collidepoint(x, y):
                        self.difficulty = DIFFICULTY_EASY
                        self.ai_type = AI_TYPE_MINIMAX
                    elif self.btn_medium.collidepoint(x, y):
                        self.difficulty = DIFFICULTY_MEDIUM
                        self.ai_type = AI_TYPE_MINIMAX
                    elif self.btn_hard.collidepoint(x, y):
                        self.difficulty = DIFFICULTY_HARD
                        self.ai_type = AI_TYPE_MINIMAX
                    # 训练AI
                    elif self.btn_train_ai.collidepoint(x, y) and HAS_TORCH:
                        self.show_training = True
                        self.training_interface.show()
                        self._run_training()
                        self.show_training = False
                        pygame.display.set_mode((WINDOW_SIZE, WINDOW_HEIGHT))
                        pygame.display.set_caption("Gomoku - 五子棋 AI 训练系统")
                    # 游戏模式
                    elif self.btn_pvp.collidepoint(x, y):
                        self.mode = 'pvp'
                        return
                    elif self.btn_pve.collidepoint(x, y):
                        self.mode = 'pve'
                        return
            self.draw_menu()
            pygame.time.Clock().tick(30)
    
    def _run_training(self):
        """运行训练界面"""
        pygame.display.set_mode((TRAINING_WIDTH, TRAINING_HEIGHT))
        
        while self.training_interface.visible:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.training_interface.handle_click(event.pos)
            
            self.training_interface.draw(pygame.display.get_surface())
            pygame.display.flip()
            pygame.time.Clock().tick(30)
        
        # 如果点击了对战AI按钮，返回游戏
        if self.return_to_game:
            self.return_to_game = False
            return

    def reset_game(self):
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = BLACK_STONE
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.ai_result = None
        self.move_history = []
        self.last_move = None
        self.board_needs_redraw = True
        self.info_needs_redraw = True
        self.hint_position = None
        self.hint_thinking = False

    def swap_sides(self):
        if self.mode != 'pve':
            return
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = WHITE_STONE if self.current_player == BLACK_STONE else BLACK_STONE
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.ai_result = None
        self.move_history = []
        self.last_move = None
        self.hint_position = None
        self.hint_thinking = False
        self.board_needs_redraw = True
        self.info_needs_redraw = True
        if not self.game_over and self.mode == 'pve' and self.current_player == WHITE_STONE:
            self.ai_move()

    def undo_move(self, steps=1):
        if self.game_over or len(self.move_history) == 0 or self.ai_thinking:
            return
        
        if self.mode == 'pve':
            steps = min(2, len(self.move_history))
        
        for _ in range(steps):
            if len(self.move_history) == 0:
                break
            i, j, player = self.move_history.pop()
            self.board[i][j] = EMPTY
            self.current_player = player
        
        self.last_move = self.move_history[-1][:2] if self.move_history else None
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.ai_result = None
        self.hint_position = None
        self.board_needs_redraw = True
        self.info_needs_redraw = True

    def draw_board(self):
        needs_full_redraw = self.board_needs_redraw or self.info_needs_redraw
        
        if needs_full_redraw:
            self.buffer.fill(BOARD_BG)
            
            # 信息栏
            pygame.draw.rect(self.buffer, INFO_BG, (0, 0, WINDOW_SIZE, INFO_HEIGHT))
            pygame.draw.line(self.buffer, (60, 60, 70), (0, INFO_HEIGHT), (WINDOW_SIZE, INFO_HEIGHT), 2)
            
            mouse_pos = pygame.mouse.get_pos()
            
            # ===== 信息栏左侧：游戏信息 =====
            move_count = len(self.move_history)
            text = self.tiny_font.render(f"Moves: {move_count}", True, INFO_TEXT)
            self.buffer.blit(text, (15, 5))
            
            if self.mode == 'pve':
                diff_names = {DIFFICULTY_EASY: "Easy", DIFFICULTY_MEDIUM: "Medium", DIFFICULTY_HARD: "Hard", DIFFICULTY_DL: "Deep AI"}
                ai_type_str = "Minimax" if self.ai_type == AI_TYPE_MINIMAX else "Neural Net"
                text = self.tiny_font.render(f"{diff_names.get(self.difficulty, 'Medium')} ({ai_type_str})", True, INFO_TEXT)
                self.buffer.blit(text, (15, 20))
            
            # 状态文字
            if self.game_over:
                if self.winner == BLACK_STONE:
                    status_text = "Black Wins!" if self.mode == 'pve' else "Black Wins!"
                elif self.winner == WHITE_STONE:
                    status_text = "White Wins!" if self.mode == 'pve' else "White Wins!"
                else:
                    status_text = "Draw"
                color = WIN_COLOR
                stone_color = (100, 100, 100)
            else:
                if self.mode == 'pvp':
                    status_text = "Black's Turn" if self.current_player == BLACK_STONE else "White's Turn"
                    stone_color = BLACK if self.current_player == BLACK_STONE else WHITE
                else:
                    status_text = "Your Turn" if self.current_player == BLACK_STONE else "AI Thinking..."
                    stone_color = BLACK if self.current_player == BLACK_STONE else (150, 150, 150)
            
            # 当前玩家指示器
            pygame.draw.circle(self.buffer, stone_color, (WINDOW_SIZE//2 - 70, 35), 8)
            if self.current_player == WHITE_STONE or self.mode == 'pvp':
                pygame.draw.circle(self.buffer, (100,100,100), (WINDOW_SIZE//2 - 70, 35), 8, 2)
            
            text = self.font.render(status_text, True, color if self.game_over else INFO_TEXT)
            self.buffer.blit(text, (WINDOW_SIZE//2 - 55, 22))
            
            # AI思考动画
            if self.ai_thinking:
                thinking_text = self.tiny_font.render("Thinking...", True, (150, 150, 150))
                self.buffer.blit(thinking_text, (WINDOW_SIZE//2 + 100, 28))
            
            # ===== 按钮组（右侧） =====
            # Menu按钮
            is_hover = self.btn_menu.collidepoint(mouse_pos)
            color = BUTTON_HOVER if is_hover else BUTTON_COLOR
            pygame.draw.rect(self.buffer, color, self.btn_menu, border_radius=6)
            text = self.small_font.render("Menu", True, (255,255,255))
            self.buffer.blit(text, (self.btn_menu.x + (self.btn_menu.width - text.get_width())//2, self.btn_menu.y + 9))
            
            # Undo按钮
            can_undo = len(self.move_history) > 0 and not self.ai_thinking and not self.game_over
            is_hover = self.btn_undo.collidepoint(mouse_pos) and can_undo
            color = BUTTON_HOVER if (is_hover and can_undo) else (BUTTON_COLOR if can_undo else (50, 50, 60))
            pygame.draw.rect(self.buffer, color, self.btn_undo, border_radius=6)
            text = self.small_font.render("Undo", True, (180,180,180) if not can_undo else (255,255,255))
            self.buffer.blit(text, (self.btn_undo.x + (self.btn_undo.width - text.get_width())//2, self.btn_undo.y + 9))
            
            # Sound按钮
            is_hover = self.btn_sound.collidepoint(mouse_pos)
            color = BUTTON_HOVER if is_hover else BUTTON_COLOR
            pygame.draw.rect(self.buffer, color, self.btn_sound, border_radius=6)
            sound_icon = "On" if self.sound_enabled else "Off"
            text = self.small_font.render(sound_icon, True, ACCENT_COLOR if self.sound_enabled else (100,100,100))
            self.buffer.blit(text, (self.btn_sound.x + (self.btn_sound.width - text.get_width())//2, self.btn_sound.y + 9))
            
            # Hint按钮（仅PVE且玩家回合显示）
            if self.mode == 'pve' and self.current_player == BLACK_STONE and not self.game_over:
                is_hover = self.btn_hint.collidepoint(mouse_pos)
                color = BUTTON_HOVER if is_hover else BUTTON_COLOR
                pygame.draw.rect(self.buffer, color, self.btn_hint, border_radius=6)
                text = self.small_font.render("Hint", True, ACCENT_COLOR if self.hint_enabled else (200,200,200))
                self.buffer.blit(text, (self.btn_hint.x + (self.btn_hint.width - text.get_width())//2, self.btn_hint.y + 9))
            
            # Restart按钮
            is_hover = self.btn_restart.collidepoint(mouse_pos)
            color = BUTTON_HOVER if is_hover else BUTTON_COLOR
            pygame.draw.rect(self.buffer, color, self.btn_restart, border_radius=6)
            text = self.small_font.render("New", True, (255,255,255))
            self.buffer.blit(text, (self.btn_restart.x + (self.btn_restart.width - text.get_width())//2, self.btn_restart.y + 9))
            
            # Swap按钮（仅PVE）
            if self.mode == 'pve':
                is_hover = self.btn_swap.collidepoint(mouse_pos)
                color = BUTTON_HOVER if is_hover else BUTTON_COLOR
                pygame.draw.rect(self.buffer, color, self.btn_swap, border_radius=6)
                text = self.small_font.render("Swap", True, (255,255,255))
                self.buffer.blit(text, (self.btn_swap.x + (self.btn_swap.width - text.get_width())//2, self.btn_swap.y + 9))
            
            # ===== 棋盘区域 =====
            board_top = INFO_HEIGHT
            
            # 木质纹理 - 15x15棋盘区域
            for i in range(BOARD_SIZE):
                shade = int(10 * (i / BOARD_SIZE - 0.5))
                row_color = tuple(max(0, min(255, c + shade)) for c in BOARD_BG)
                pygame.draw.rect(self.buffer, row_color, 
                    (MARGIN, board_top + MARGIN + i * CELL_SIZE, 
                     (BOARD_SIZE - 1) * CELL_SIZE, CELL_SIZE))
            
            # 棋盘网格 - 15x15的网格线（15条线围成15x15格子）
            for i in range(BOARD_SIZE):
                start_pos = (MARGIN, board_top + MARGIN + i * CELL_SIZE)
                end_pos = (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, board_top + MARGIN + i * CELL_SIZE)
                pygame.draw.line(self.buffer, LINE_COLOR, start_pos, end_pos, 1)
                start_pos = (MARGIN + i * CELL_SIZE, board_top + MARGIN)
                end_pos = (MARGIN + i * CELL_SIZE, board_top + MARGIN + (BOARD_SIZE - 1) * CELL_SIZE)
                pygame.draw.line(self.buffer, LINE_COLOR, start_pos, end_pos, 1)
            
            # 棋盘边框 - 完整的15x15边框
            pygame.draw.rect(self.buffer, LINE_COLOR, 
                (MARGIN, board_top + MARGIN, 
                 (BOARD_SIZE - 1) * CELL_SIZE, (BOARD_SIZE - 1) * CELL_SIZE), 2)
            
            # 星位
            stars = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
            for si, sj in stars:
                x = MARGIN + sj * CELL_SIZE
                y = board_top + MARGIN + si * CELL_SIZE
                pygame.draw.circle(self.buffer, LINE_COLOR, (x, y), 4)
            
            # 最后落子高亮
            if self.last_move and not self.game_over:
                li, lj = self.last_move
                x = MARGIN + lj * CELL_SIZE
                y = board_top + MARGIN + li * CELL_SIZE
                pygame.draw.circle(self.buffer, (255, 220, 100), (x, y), 18, 3)
            
            # AI提示位置
            if self.hint_enabled and self.hint_position and self.mode == 'pve' and not self.game_over:
                hi, hj = self.hint_position
                x = MARGIN + hj * CELL_SIZE
                y = board_top + MARGIN + hi * CELL_SIZE
                if self.board[hi][hj] == EMPTY:
                    if self.hint_thinking:
                        pygame.draw.circle(self.buffer, (100, 255, 100), (x, y), 15, 3)
                    else:
                        pygame.draw.circle(self.buffer, (100, 255, 100), (x, y), 15, 2)
            
            # 棋子
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] != EMPTY:
                        x = MARGIN + j * CELL_SIZE
                        y = board_top + MARGIN + i * CELL_SIZE
                        # 阴影
                        pygame.draw.circle(self.buffer, (100, 80, 60), (x + 2, y + 2), 15)
                        if self.board[i][j] == BLACK_STONE:
                            pygame.draw.circle(self.buffer, BLACK, (x, y), 15)
                            pygame.draw.circle(self.buffer, (60, 60, 60), (x - 3, y - 3), 5)
                        else:
                            pygame.draw.circle(self.buffer, WHITE, (x, y), 15)
                            pygame.draw.circle(self.buffer, (180, 180, 180), (x, y), 15, 1)
                            pygame.draw.circle(self.buffer, (255, 255, 255), (x - 3, y - 3), 5)
            
            # 获胜连线
            if self.game_over and self.winner:
                line = self._get_win_line()
                if line:
                    start, end = line
                    sx = MARGIN + start[1] * CELL_SIZE
                    sy = board_top + MARGIN + start[0] * CELL_SIZE
                    ex = MARGIN + end[1] * CELL_SIZE
                    ey = board_top + MARGIN + end[0] * CELL_SIZE
                    for width in [6, 4, 2]:
                        color = WIN_COLOR if self.winner == BLACK_STONE else (255, 200, 200)
                        pygame.draw.line(self.buffer, color, (sx, sy), (ex, ey), width)
            
            self.board_needs_redraw = False
            self.info_needs_redraw = False
        
        self.screen.blit(self.buffer, (0, 0))
    
    def _get_win_line(self):
        if not self.winner:
            return None
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.winner:
                    for dx, dy in directions:
                        count = 1
                        positions = [(i, j)]
                        x, y = i+dx, j+dy
                        while 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==self.winner:
                            count += 1
                            positions.append((x, y))
                            x += dx; y += dy
                        if count >= 5:
                            return (positions[0], positions[-1])
        return None

    def screen_to_board(self, x, y):
        board_y = y - INFO_HEIGHT
        i = round((board_y - MARGIN) / CELL_SIZE)
        j = round((x - MARGIN) / CELL_SIZE)
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            return i, j
        return None, None

    def check_winner(self, i, j, player):
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for dx, dy in directions:
            count = 1
            x, y = i+dx, j+dy
            while 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==player:
                count+=1
                x+=dx; y+=dy
            x, y = i-dx, j-dy
            while 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==player:
                count+=1
                x-=dx; y-=dy
            if count>=5:
                return True
        return False

    def place_stone(self, i, j, player=None):
        if player is None:
            player = self.current_player
        if self.board[i][j] != EMPTY or self.game_over:
            return False
        self.board[i][j] = player
        self.move_history.append((i, j, player))
        self.last_move = (i, j)
        if self.sound_enabled:
            play_stone_sound()
        if self.check_winner(i, j, player):
            self.game_over = True
            self.winner = player
            if self.sound_enabled:
                play_win_sound()
            return True
        self.current_player = 3 - player
        return True

    # ---------- 评估函数 ----------
    def get_pattern_score(self, count, left_empty, right_empty):
        if count >= 5:
            return SCORES['FIVE']
        if count == 4:
            if left_empty and right_empty:
                return SCORES['ALIVE_FOUR']
            elif left_empty or right_empty:
                return SCORES['SLEEP_FOUR']
        if count == 3:
            if left_empty and right_empty:
                return SCORES['ALIVE_THREE']
            elif left_empty or right_empty:
                return SCORES['SLEEP_THREE']
        if count == 2:
            if left_empty and right_empty:
                return SCORES['ALIVE_TWO']
            elif left_empty or right_empty:
                return SCORES['SLEEP_TWO']
        return 0

    def evaluate_direction(self, i, j, dx, dy, player):
        count = 1
        x, y = i+dx, j+dy
        while 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==player:
            count+=1
            x+=dx; y+=dy
        right_empty = (0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==EMPTY)
        x, y = i-dx, j-dy
        while 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==player:
            count+=1
            x-=dx; y-=dy
        left_empty = (0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and self.board[x][y]==EMPTY)
        return self.get_pattern_score(count, left_empty, right_empty)

    def evaluate_position(self, i, j, player):
        if self.board[i][j] != EMPTY:
            return 0
        self.board[i][j] = player
        score = 0
        for dx, dy in [(0,1),(1,0),(1,1),(1,-1)]:
            score += self.evaluate_direction(i, j, dx, dy, player)
        self.board[i][j] = EMPTY
        return score

    def evaluate_board(self, player):
        my_score = 0
        opp_score = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == EMPTY:
                    my_score += self.evaluate_position(i, j, player)
                    opp_score += self.evaluate_position(i, j, 3 - player)
        return my_score - 2.5 * opp_score

    def get_candidate_moves(self):
        candidates = set()
        empty_count = sum(row.count(EMPTY) for row in self.board)
        if empty_count == BOARD_SIZE * BOARD_SIZE:
            return [(BOARD_SIZE//2, BOARD_SIZE//2)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] != EMPTY:
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i+di, j+dj
                            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and self.board[ni][nj] == EMPTY:
                                candidates.add((ni, nj))
        if not candidates:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == EMPTY:
                        candidates.add((i, j))
        return list(candidates)

    def minimax(self, depth, alpha, beta, is_maximizing, ai_player):
        if depth == 0 or self.game_over:
            return self.evaluate_board(ai_player)
        candidates = self.get_candidate_moves()
        if not candidates:
            return 0
        if is_maximizing:
            max_eval = -float('inf')
            for i, j in candidates:
                self.board[i][j] = ai_player
                if self.check_winner(i, j, ai_player):
                    self.board[i][j] = EMPTY
                    return 1000000
                eval_score = self.minimax(depth-1, alpha, beta, False, ai_player)
                self.board[i][j] = EMPTY
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - ai_player
            for i, j in candidates:
                self.board[i][j] = opponent
                if self.check_winner(i, j, opponent):
                    self.board[i][j] = EMPTY
                    return -1000000
                eval_score = self.minimax(depth-1, alpha, beta, True, ai_player)
                self.board[i][j] = EMPTY
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move_minimax(self, depth=3):
        candidates = self.get_candidate_moves()
        if not candidates:
            return None
        opponent = 3 - self.current_player
        # 直接防守胜利
        for i, j in candidates:
            self.board[i][j] = opponent
            if self.check_winner(i, j, opponent):
                self.board[i][j] = EMPTY
                return (i, j)
            self.board[i][j] = EMPTY
        # 防守活三/冲四
        urgent = []
        for i, j in candidates:
            score_opp = self.evaluate_position(i, j, opponent)
            if score_opp >= 4000:
                urgent.append((score_opp, i, j))
        if urgent:
            urgent.sort(reverse=True)
            return (urgent[0][1], urgent[0][2])
        # 进攻排序
        move_scores = []
        for i, j in candidates:
            score = self.evaluate_position(i, j, self.current_player)
            move_scores.append((score, i, j))
        move_scores.sort(reverse=True)
        best_score = -float('inf')
        best_move = None
        for _, i, j in move_scores[:15]:
            self.board[i][j] = self.current_player
            if self.check_winner(i, j, self.current_player):
                self.board[i][j] = EMPTY
                return (i, j)
            eval_score = self.minimax(depth-1, -float('inf'), float('inf'), False, self.current_player)
            self.board[i][j] = EMPTY
            if eval_score > best_score:
                best_score = eval_score
                best_move = (i, j)
        return best_move

    def ai_move(self, depth=None):
        if self.game_over or self.current_player != WHITE_STONE or self.mode != 'pve':
            return
        if self.ai_thinking:
            return
        if depth is None:
            depth = DIFFICULTY_DEPTHS.get(self.difficulty, 3)
        self.ai_thinking = True
        self.ai_result = None
        
        def compute():
            if self.ai_type == AI_TYPE_DL and self.dl_net is not None:
                move = self._get_dl_move()
            else:
                move = self.get_best_move_minimax(depth)
            with self.ai_lock:
                self.ai_result = move
            self.ai_thinking = False
        
        threading.Thread(target=compute, daemon=True).start()
    
    def _get_dl_move(self):
        """使用深度学习网络获取落子"""
        valid_moves = self.get_candidate_moves()
        if not valid_moves:
            return None
        state = self.get_state()
        return self.dl_net.get_action(state, valid_moves)

    # ---------- 主循环 ----------
    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.run_menu()
            self.reset_game()
            self.return_to_menu = False
            
            while not self.return_to_menu:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if self.btn_menu.collidepoint(x, y):
                            self.return_to_menu = True
                        elif self.btn_undo.collidepoint(x, y):
                            self.undo_move()
                        elif self.btn_restart.collidepoint(x, y):
                            self.reset_game()
                            if self.mode == 'pve' and not self.game_over and self.current_player == WHITE_STONE:
                                self.ai_move()
                        elif self.btn_sound.collidepoint(x, y):
                            self.sound_enabled = not self.sound_enabled
                            self.info_needs_redraw = True
                        elif self.btn_hint.collidepoint(x, y) and self.mode == 'pve':
                            if self.hint_enabled:
                                self.hint_enabled = False
                                self.hint_position = None
                            else:
                                self.hint_enabled = True
                                self.hint_position = None
                                self.hint_thinking = True
                                self.info_needs_redraw = True
                                # 计算提示位置
                                import copy
                                board_backup = copy.deepcopy(self.board)
                                self.hint_position = self.get_best_move_minimax(DIFFICULTY_DEPTHS.get(self.difficulty, 2))
                                self.board = board_backup
                                self.hint_thinking = False
                        elif self.mode == 'pve' and self.btn_swap.collidepoint(x, y):
                            self.swap_sides()
                        else:
                            i, j = self.screen_to_board(x, y)
                            if i is not None and not self.game_over:
                                if self.mode == 'pvp':
                                    if self.place_stone(i, j):
                                        self.board_needs_redraw = True
                                        self.info_needs_redraw = True
                                elif self.mode == 'pve' and self.current_player == BLACK_STONE:
                                    if self.place_stone(i, j):
                                        self.board_needs_redraw = True
                                        self.info_needs_redraw = True
                                        if not self.game_over:
                                            self.ai_move()
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_u:
                            self.undo_move()
                        elif event.key == pygame.K_r:
                            self.reset_game()
                            if self.mode == 'pve' and not self.game_over and self.current_player == WHITE_STONE:
                                self.ai_move()
                        elif event.key == pygame.K_ESCAPE:
                            self.return_to_menu = True
                        elif event.key == pygame.K_s and self.mode == 'pve':
                            self.swap_sides()
                
                # 处理 AI 计算结果
                if self.ai_result is not None and not self.ai_thinking:
                    move = self.ai_result
                    self.ai_result = None
                    if move and not self.game_over:
                        i, j = move
                        self.board[i][j] = WHITE_STONE
                        self.move_history.append((i, j, WHITE_STONE))
                        self.last_move = (i, j)
                        if self.sound_enabled:
                            play_stone_sound()
                        self.board_needs_redraw = True
                        self.info_needs_redraw = True
                        if self.check_winner(i, j, WHITE_STONE):
                            self.game_over = True
                            self.winner = WHITE_STONE
                            if self.sound_enabled:
                                play_win_sound()
                        else:
                            self.current_player = BLACK_STONE
                
                # 绘制
                self.draw_board()
                pygame.display.flip()
                clock.tick(60)

if __name__ == "__main__":
    game = Gomoku()
    game.run()
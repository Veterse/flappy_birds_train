import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import cv2

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(FlappyBirdEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: ничего, 1: прыжок
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.render_mode = render_mode
        self.screen_width = 288
        self.screen_height = 512
        self.clock = None
        self._init_pygame()
        self.reset()

    def _init_pygame(self):
        pygame.init()
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        pygame.display.set_caption('Flappy Bird RL')
        self.clock = pygame.time.Clock()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.bird_y = self.screen_height // 2
        self.bird_vel = 0
        self.gravity = 1.5
        self.flap_power = -10
        self.bird_x = 50
        self.pipe_gap = int(140 * 1.2)
        self.pipe_width = 52
        self.pipe_vel = -4
        self.score = 0
        self.done = False
        self.pipes = []
        self.frame_count = 0
        self.pipe_min_distance = 150  # Минимальное расстояние между трубами (ещё меньше)
        self._add_pipe()
        self.passed_pipe = False
        return self._get_observation(), {}

    def _add_pipe(self):
        gap_y = np.random.randint(60, self.screen_height - 60 - self.pipe_gap)
        pipe = {
            'x': self.screen_width,
            'gap_y': gap_y
        }
        self.pipes.append(pipe)

    def step(self, action):
        reward = 0.01  # Очень маленькая награда за каждый кадр в воздухе
        info = {}
        if action == 1:
            self.bird_vel = self.flap_power
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        # Move pipes
        for pipe in self.pipes:
            pipe['x'] += self.pipe_vel
        # Remove passed pipes
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
        # Если после удаления труб нет — добавить новую
        if not self.pipes:
            self._add_pipe()
        # Add new pipe, если последняя труба далеко
        if self.pipes[-1]['x'] < self.screen_width - self.pipe_min_distance:
            self._add_pipe()
        # Check if passed pipe (bird_x > pipe_x + pipe_width)
        passed = False
        if self.pipes[0]['x'] + self.pipe_width < self.bird_x and not self.passed_pipe:
            reward += 1  # +1 за прохождение трубы
            self.score += 1
            passed = True
            self.passed_pipe = True
        if self.pipes[0]['x'] + self.pipe_width >= self.bird_x:
            self.passed_pipe = False
        # Collision
        self.done = self._check_collision()
        terminated = self.done
        truncated = False  # Можно добавить лимит по шагам, если нужно
        if self.done:
            reward = -100
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _check_collision(self):
        # Ground/ceiling
        if self.bird_y > self.screen_height - 24 or self.bird_y < 0:
            return True
        # Pipes
        bird_rect = pygame.Rect(self.bird_x, int(self.bird_y), 34, 24)
        for pipe in self.pipes:
            # Top pipe
            top_rect = pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['gap_y'])
            # Bottom pipe
            bottom_rect = pygame.Rect(pipe['x'], pipe['gap_y'] + self.pipe_gap, self.pipe_width, self.screen_height - (pipe['gap_y'] + self.pipe_gap))
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
        return False

    def _draw(self):
        self.screen.fill((135, 206, 235))  # Sky blue
        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe['x'], 0, self.pipe_width, pipe['gap_y']))
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe['x'], pipe['gap_y'] + self.pipe_gap, self.pipe_width, self.screen_height - (pipe['gap_y'] + self.pipe_gap)))
        # Draw bird
        pygame.draw.rect(self.screen, (255, 255, 0), (self.bird_x, int(self.bird_y), 34, 24))
        # Draw ground
        pygame.draw.rect(self.screen, (222, 184, 135), (0, self.screen_height - 24, self.screen_width, 24))
        if self.render_mode:
            pygame.display.flip()
            self.clock.tick(30)

    def _get_observation(self):
        self._draw()
        raw = pygame.surfarray.array3d(self.screen)
        img = cv2.cvtColor(np.transpose(raw, (1, 0, 2)), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=-1)
        return img.astype(np.uint8)

    def render(self, mode='human'):
        if not self.render_mode:
            self.render_mode = True
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self._draw()

    def close(self):
        pygame.quit()

    def seed(self, seed=None):
        np.random.seed(seed) 
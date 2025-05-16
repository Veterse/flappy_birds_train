from flappy_env import FlappyBirdEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch
import numpy as np
import cv2
import glob
import os
import re
import shutil

CHECKPOINT_DIR = 'E:/flappy_checkpoints/'

class RewardClipper(VecEnvWrapper):
    def __init__(self, venv, min_reward=-1.0, max_reward=1.0):
        super().__init__(venv)
        self.min_reward = min_reward
        self.max_reward = max_reward
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.clip(rewards, self.min_reward, self.max_reward)
        return obs, rewards, dones, infos
    def reset(self):
        return self.venv.reset()

def make_env(render_mode=False):
    return Monitor(FlappyBirdEnv(render_mode=render_mode))

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=10000, verbose=1):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _run_demo(self, deterministic, label):
        env = make_env(render_mode=True)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        try:
            while not done:
                obs_input = np.transpose(obs, (2, 0, 1))
                obs_input = np.expand_dims(obs_input, axis=0)
                with torch.no_grad():
                    action, _ = self.model.predict(obs_input, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render()
                try:
                    cv2.imshow(f"Agent view (84x84) [{label}]", obs)
                except cv2.error:
                    pass
                if cv2.waitKey(10) == 27:
                    break
                done = terminated or truncated
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            env.close()
        print(f"[RenderCallback] Demo episode reward ({label}): {total_reward}")

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self._run_demo(deterministic=True, label="deterministic")
            self._run_demo(deterministic=False, label="exploration")
        return True

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, print_freq=10, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        self.episode_count = 0

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
                    self.episode_count += 1
                    if reward > self.best_reward:
                        self.best_reward = reward
                    if len(self.episode_rewards) % self.print_freq == 0:
                        avg_reward = sum(self.episode_rewards[-self.print_freq:]) / self.print_freq
                        avg_length = sum(self.episode_lengths[-self.print_freq:]) / self.print_freq
                        print(f'[EpisodeStats] Эпизод: {self.episode_count} | Последняя награда: {reward:.2f} | Лучшая награда: {self.best_reward:.2f} | Средняя награда за {self.print_freq}: {avg_reward:.2f} | Средняя длина: {avg_length:.2f} | Timesteps: {self.model.num_timesteps}')
        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            avg_total = sum(self.episode_rewards) / len(self.episode_rewards)
            print(f'\n[EpisodeStats] Итог: всего эпизодов: {self.episode_count}, лучшая награда: {self.best_reward:.2f}, средняя награда: {avg_total:.2f}')

def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    files = glob.glob(os.path.join(checkpoint_dir, 'dqn_flappybird_*_steps.zip'))
    pattern = re.compile(r'dqn_flappybird_(\d+)_steps.zip')
    files_steps = [(f, int(pattern.search(os.path.basename(f)).group(1)))
                  for f in files if pattern.search(os.path.basename(f))]
    if not files_steps:
        return None
    latest = max(files_steps, key=lambda x: x[1])[0]
    return latest

def check_model_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def list_named_checkpoints(checkpoint_dir=CHECKPOINT_DIR):
    files = glob.glob(os.path.join(checkpoint_dir, 'dqn_flappybird_*.zip'))
    files = sorted(files, key=os.path.getmtime)
    return files

def prompt_checkpoint_selection(files):
    print("\nДоступные чекпоинты:")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    idx = input(f"Введите номер чекпоинта для продолжения (0-{len(files)-1}, Enter для последнего): ")
    if idx.strip() == '':
        return files[-1]
    try:
        idx = int(idx)
        if 0 <= idx < len(files):
            return files[idx]
    except Exception:
        pass
    print("Некорректный ввод, выбран последний чекпоинт.")
    return files[-1]

def save_checkpoint_with_buffer(model, path):
    model.save(path)
    rb_path = path.replace('.zip', '_rb.pkl')
    try:
        model.save_replay_buffer(rb_path)
        print(f"[INFO] Replay buffer сохранён как {rb_path}")
    except Exception as e:
        print(f"[WARNING] Не удалось сохранить replay buffer: {e}")
    # Сохраняем exploration_rate
    exp_path = path.replace('.zip', '_exploration.txt')
    try:
        with open(exp_path, 'w') as f:
            f.write(str(model.exploration_rate))
        print(f"[INFO] exploration_rate сохранён как {exp_path}")
    except Exception as e:
        print(f"[WARNING] Не удалось сохранить exploration_rate: {e}")

def load_checkpoint_with_buffer(path, env, device):
    model = DQN.load(path, env=env, device=device)
    rb_path = path.replace('.zip', '_rb.pkl')
    if os.path.exists(rb_path):
        try:
            model.load_replay_buffer(rb_path)
            print(f"[INFO] Replay buffer загружен из {rb_path}")
        except Exception as e:
            print(f"[WARNING] Не удалось загрузить replay buffer: {e}")
    else:
        print(f"[WARNING] Replay buffer не найден для {path}")
    print(f"[INFO] Продолжаю обучение с буфером: {rb_path if os.path.exists(rb_path) else 'ПУСТОЙ буфер'}")
    try:
        print(f"[DEBUG] Размер replay buffer после загрузки: {model.replay_buffer.size()}")
    except Exception as e:
        print(f"[DEBUG] Не удалось получить размер replay buffer: {e}")
    # Восстанавливаем exploration_rate
    exp_path = path.replace('.zip', '_exploration.txt')
    if os.path.exists(exp_path):
        try:
            with open(exp_path, 'r') as f:
                model.exploration_rate = float(f.read())
            print(f"[INFO] Восстановлен exploration_rate: {model.exploration_rate}")
        except Exception as e:
            print(f"[WARNING] Не удалось восстановить exploration_rate: {e}")
    else:
        print(f"[WARNING] exploration_rate не найден для {path}")
    return model

if __name__ == "__main__":
    # Включаем оптимизацию cuDNN для ускорения на GPU
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Используется устройство: {device}")
    # Создаём папки для чекпоинтов и логов
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    # Для обучения
    env = DummyVecEnv([lambda: make_env(render_mode=False) for _ in range(12)])
    env = VecTransposeImage(env)
    env = RewardClipper(env, min_reward=-1.0, max_reward=1.0)
    # TensorBoard логирование
    new_logger = configure('./logs/tb', ['stdout', 'tensorboard'])
    named_checkpoints = list_named_checkpoints(CHECKPOINT_DIR)
    if named_checkpoints:
        print("[INFO] Найдены чекпоинты в папке checkpoints.")
        checkpoint_path = prompt_checkpoint_selection(named_checkpoints)
        print(f"[INFO] Загружаю чекпоинт: {checkpoint_path}")
        model = load_checkpoint_with_buffer(checkpoint_path, env=env, device=device)
        model.set_logger(new_logger)
        print(f"[INFO] Используется устройство: {model.device}")
        print(f"[INFO] Старт обучения с {checkpoint_path}, шагов: {getattr(model, 'num_timesteps', 'неизвестно')}")
    else:
        print("[INFO] Чекпоинты не найдены. Начинаю обучение с нуля.")
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1.5e-3,
            buffer_size=200_000,
            learning_starts=1000,
            batch_size=5985,
            train_freq=2,
            target_update_interval=500,
            verbose=1,
            device=device,
            exploration_fraction=0.3,
            exploration_final_eps=0.01,
            tensorboard_log="./logs/tb"
        )
        model.set_logger(new_logger)
        print(f"[INFO] Используется устройство: {model.device}")
        print(f"[INFO] Старт обучения с нуля.")
    # Для оценки
    eval_env = DummyVecEnv([lambda: make_env(render_mode=False) for _ in range(12)])
    eval_env = VecTransposeImage(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        log_path="./logs/",
        eval_freq=50000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1
    )
    render_callback = RenderCallback(render_freq=50000)
    episode_stats_callback = EpisodeStatsCallback(print_freq=10)
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[eval_callback, render_callback, episode_stats_callback],
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("[WARNING] Обучение прервано пользователем! Сохраняю текущую модель...")
        name = input("Введите имя или номер для нового чекпоинта (например, 2): ").strip()
        if not name:
            name = "manual"
        path = f"{CHECKPOINT_DIR}dqn_flappybird_{name}.zip"
        save_checkpoint_with_buffer(model, path)
        print(f"[INFO] Модель и replay buffer сохранены как {path}")
    except Exception as e:
        print(f"[ERROR] Обучение завершилось с ошибкой: {e}\nСохраняю текущую модель...")
        name = input("Введите имя или номер для нового чекпоинта (например, 2): ").strip()
        if not name:
            name = "error"
        path = f"{CHECKPOINT_DIR}dqn_flappybird_{name}.zip"
        save_checkpoint_with_buffer(model, path)
        print(f"[INFO] Модель и replay buffer сохранены как {path}")
        raise
    else:
        print("[INFO] Обучение завершено успешно.")
        name = input("Введите имя или номер для нового чекпоинта (например, 2): ").strip()
        if not name:
            name = "final"
        path = f"{CHECKPOINT_DIR}dqn_flappybird_{name}.zip"
        save_checkpoint_with_buffer(model, path)
        print(f"[INFO] Финальная модель и replay buffer сохранены как {path}")
    finally:
        env.close()
        eval_env.close()
        # Удаляю сообщения о best_model.zip
        last_checkpoint = find_latest_checkpoint()
        if last_checkpoint:
            print(f"[INFO] Последний чекпоинт: {last_checkpoint}")
        else:
            print(f"[WARNING] Чекпоинты не найдены!") 
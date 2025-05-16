from flappy_env import FlappyBirdEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np
import cv2
import glob
import os

CHECKPOINT_DIR = 'E:/flappy_checkpoints/'

def load_checkpoint_with_buffer(path, env, device=None):
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    files = glob.glob(os.path.join(checkpoint_dir, 'dqn_flappybird_*.zip'))
    if not files:
        return None
    files = sorted(files, key=os.path.getmtime)
    return files[-1]

if __name__ == "__main__":
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        print(f"[ERROR] Нет чекпоинтов в папке {CHECKPOINT_DIR}! Сначала обучите агента.")
        exit(1)
    print(f"[INFO] Загружаю последний чекпоинт: {checkpoint_path}")
    env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode=True)])
    env = VecTransposeImage(env)
    model = load_checkpoint_with_buffer(checkpoint_path, env=env)
    obs = env.reset()
    total_reward = 0
    while True:
        obs_input = obs  # уже (1, 1, 84, 84) после VecTransposeImage
        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.envs[0].render()  # рендерим оригинальную среду внутри VecEnv
        try:
            cv2.imshow("Agent view (84x84)", obs[0].transpose(1, 2, 0))
        except cv2.error:
            pass
        if cv2.waitKey(10) == 27:
            break
        if done:
            print(f"Episode reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
    cv2.destroyAllWindows()
    env.close()
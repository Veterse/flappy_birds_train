from flappy_env import FlappyBirdEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
import torch
import os
import numpy as np
import cv2

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=10000, verbose=1):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            env = FlappyBirdEnv(render_mode=True)
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                obs_input = np.transpose(obs, (2, 0, 1))
                obs_input = np.expand_dims(obs_input, axis=0)
                action, _ = self.model.predict(obs_input, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render()
                cv2.imshow("Agent view (84x84)", obs)
                if cv2.waitKey(10) == 27:
                    break
                done = terminated or truncated
            print(f"[RenderCallback] Demo episode reward: {total_reward}")
            cv2.destroyAllWindows()
            env.close()
        return True

if __name__ == "__main__":
    # Создание среды
    env = FlappyBirdEnv(render_mode=False)

    # Проверка среды на совместимость с Gym API
    check_env(env, warn=True)

    # Настройка DQN с CNN-политикой
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1.5e-3,
        buffer_size=300000,
        learning_starts=1000,
        batch_size=984,
        train_freq=8,
        target_update_interval=1000,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Используется устройство: {model.device}")

    # Callback для оценки и логирования
    eval_env = FlappyBirdEnv(render_mode=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    render_callback = RenderCallback(render_freq=10000)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./checkpoints/',
        name_prefix='dqn_flappybird'
    )

    # Обучение с логированием и автосохранением чекпоинтов
    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, render_callback, checkpoint_callback]
    )

    # Сохранение модели
    model.save("dqn_flappybird_cnn")

    env.close()
    eval_env.close() 
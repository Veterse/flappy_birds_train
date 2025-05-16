from flappy_env import FlappyBirdEnv
from stable_baselines3 import DQN
import cv2
import numpy as np

if __name__ == "__main__":
    # Загрузка обученной модели
    model = DQN.load("dqn_flappybird_cnn")

    # Создание среды с визуализацией
    env = FlappyBirdEnv(render_mode=True)
    obs = env.reset()
    done = False
    total_reward = 0

    while True:
        obs = np.array(obs)
        # Преобразуем obs: (84,84,1) -> (1,1,84,84)
        obs_input = np.transpose(obs, (2, 0, 1))  # (1,84,84)
        obs_input = np.expand_dims(obs_input, axis=0)  # (1,1,84,84)
        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        # Визуализация observation через OpenCV (грейскейл)
        cv2.imshow("Agent view (84x84)", obs)
        if cv2.waitKey(10) == 27:  # ESC для выхода
            break
        if done:
            print(f"Episode reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
    cv2.destroyAllWindows()
    env.close() 
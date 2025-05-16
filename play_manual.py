from flappy_env import FlappyBirdEnv
import pygame
import cv2

if __name__ == "__main__":
    env = FlappyBirdEnv(render_mode=True)
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while True:
        action = 0  # ничего не делать
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # прыжок
                if event.key == pygame.K_ESCAPE:
                    done = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        cv2.imshow("Manual play (84x84)", obs)
        if cv2.waitKey(10) == 27 or done:
            break
        if terminated or truncated:
            print(f"Game over! Total reward: {total_reward}")
            obs, _ = env.reset()
            total_reward = 0

    cv2.destroyAllWindows()
    env.close() 
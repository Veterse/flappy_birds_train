# Flappy Bird RL (DQN, Stable Baselines3)

Реинфорсмент-обучение агента играть в Flappy Bird с визуальным входом (84x84 grayscale) на собственной среде (pygame, совместима с Gym API). Для обучения используется DQN из Stable Baselines3.

## Установка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. (Рекомендуется) Установите PyTorch с поддержкой CUDA, если есть GPU:
https://pytorch.org/get-started/locally/

## Структура проекта
- `flappy_env.py` — RL-среда Flappy Bird (pygame, Gym API, визуальный observation)
- `train_sb3.py` — обучение агента DQN с автосохранением, логированием, визуализацией
- `continue_train.py` — продолжение обучения с последнего чекпоинта или с нуля
- `eval_agent.py` — запуск и визуализация игры обученного агента
- `watch_checkpoint.py` — просмотр игры агента с последнего чекпоинта или лучшей модели
- `play_manual.py` — ручное управление (играете сами)
- `requirements.txt` — зависимости

## Как обучить агента

```bash
python train_sb3.py
```
- Чекпоинты сохраняются каждые 20 000 шагов в папку `checkpoints/`.
- Лучшая модель по результатам оценки — `logs/best_model.zip`.
- При остановке (Ctrl+C) или ошибке модель сохраняется как `dqn_flappybird_interrupted.zip` или `dqn_flappybird_error.zip`.
- Финальная модель — `dqn_flappybird_cnn.zip`.

## Как продолжить обучение

```bash
python continue_train.py
```
- Скрипт сам найдёт последний чекпоинт и продолжит обучение.
- Можно выбрать нужный чекпоинт вручную.

## Как посмотреть игру агента

```bash
python watch_checkpoint.py
```
- Загружает `logs/best_model.zip` (если есть), иначе последний чекпоинт.
- Показывает игру агента с визуализацией.

## Как поиграть вручную

```bash
python play_manual.py
```
- Управление: пробел — прыжок, Esc — выход.

## Пример использования среды

```python
from flappy_env import FlappyBirdEnv
import cv2

env = FlappyBirdEnv(render_mode=True)
obs, _ = env.reset()
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    cv2.imshow('obs', obs)
    if cv2.waitKey(1) == 27:
        break
    if terminated or truncated:
        obs, _ = env.reset()
cv2.destroyAllWindows()
env.close()
```

## Восстановление после сбоя
- После сбоя или остановки всегда есть последний чекпоинт и/или best_model. Просто продолжайте обучение или смотрите игру через `watch_checkpoint.py`.

---

**Вопросы, баги, предложения — пишите в Issues или Pull Requests!** 
# Flappy Bird RL с YaLA

## Установка зависимостей

1. Установите основные библиотеки:

```bash
pip install -r requirements.txt
```

2. Установите YaLA из исходников:

```bash
git clone https://github.com/facebookresearch/YaLA.git
cd YaLA
pip install -r requirements.txt
pip install -e .
cd ..
```

## Структура проекта
- `flappy_env.py` — RL-среда Flappy Bird (pygame, Gym API)
- `train_yala.py` — обучение агента YaLA
- `eval_agent.py` — запуск/оценка агента, визуализация
- `utils.py` — вспомогательные функции

## Запуск среды (пример)

```python
from flappy_env import FlappyBirdEnv
import cv2

env = FlappyBirdEnv(render_mode=True)
obs = env.reset()
done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
    cv2.imshow('obs', obs)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
env.close()
```

## Как не потерять прогресс и как восстановиться

- **Автосохранение чекпоинтов**: каждые 20 000 шагов в папку `checkpoints/`.
- **Лучшая модель**: автоматически сохраняется в `logs/best_model.zip` (по результатам оценки).
- **Ручное сохранение при остановке**: если вы прервёте обучение (Ctrl+C), текущая модель сохранится как `dqn_flappybird_interrupted.zip`.
- **Если произошла ошибка**: модель сохранится как `dqn_flappybird_error.zip`.
- **Финальная модель**: после успешного завершения обучения сохраняется как `dqn_flappybird_cnn.zip`.

### Как продолжить обучение
- Просто запустите `continue_train.py` — он сам найдёт последний чекпоинт и продолжит обучение.

### Как посмотреть лучшую игру агента
- Запустите `watch_checkpoint.py` — он загрузит `logs/best_model.zip` (если есть), иначе последний чекпоинт.

### Как восстановиться после сбоя
- После сбоя или остановки всегда есть последний чекпоинт и/или best_model. Просто продолжайте обучение или смотрите игру через `watch_checkpoint.py`.

---

**Для обучения и запуска агента см. train_yala.py и eval_agent.py** 
import argparse
from pathlib import Path

import numpy as np
import pygame
import zarr


# 1. Select episode
parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--dataset", type=str, default="pusht_demo.zarr")
args = parser.parse_args()

# 2. Load dataset
dataset_path = (
    Path(__file__).resolve().parents[1]
    / "data/demonstrations" / args.dataset
)

root = zarr.open_group(str(dataset_path), mode="r")

data = root["data"]
ends = np.asarray(root["meta"]["episode_ends"][:])

episode_id = args.episode

if not 0 <= episode_id < len(ends):
    parser.error(f"Episode must be from 0 to {len(ends) - 1}")

start = 0 if episode_id == 0 else int(ends[episode_id - 1])
stop = int(ends[episode_id])

# 3. Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((960, 576))
pygame.display.set_caption("Push-T Dataset Viewer")

font = pygame.font.SysFont("consolas", 19)
clock = pygame.time.Clock()

step = 0
playing = False
running = True

# 4. Visualization loop
while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

            elif event.key == pygame.K_RIGHT:
                step = min(step + 1, stop - start - 1)

            elif event.key == pygame.K_LEFT:
                step = max(step - 1, 0)

            elif event.key == pygame.K_SPACE:
                playing = not playing

            elif event.key == pygame.K_r:
                step = 0
                playing = True

    if not running:
        break

    # 5. Read synchronized data
    idx = start + step

    img = np.asarray(data["img"][idx])
    state = np.asarray(data["state"][idx])
    action = np.asarray(data["action"][idx])
    contacts = np.asarray(data["n_contacts"][idx])

    # 6. Render image
    surface = pygame.image.frombuffer(
        img.tobytes(),
        (img.shape[1], img.shape[0]),
        "RGB"
    )

    surface = pygame.transform.scale(
        surface,
        (576, 576)
    )

    screen.fill((35, 35, 35))
    screen.blit(surface, (0, 0))

    # 7. Render state and action
    lines = [
        f"Episode: {episode_id}/{len(ends) - 1}",
        f"Step: {step + 1}/{stop - start}",
        "",
        "AGENT POSITION",
        f"X: {state[0]:.2f}",
        f"Y: {state[1]:.2f}",
        "",
        "BLOCK POSE",
        f"X: {state[2]:.2f}",
        f"Y: {state[3]:.2f}",
        f"Angle: {state[4]:.3f} rad",
        "",
        "ACTION",
        f"X: {action[0]:.2f}",
        f"Y: {action[1]:.2f}",
        f"Contacts: {contacts[0]:.0f}",
        "",
        "LEFT/RIGHT: Step",
        "SPACE: Play/Pause",
        "R: Restart | Q: Quit"
    ]

    for i, line in enumerate(lines):
        text = font.render(line, True, (240, 240, 240))
        screen.blit(text, (595, 12 + i * 28))

    pygame.display.flip()

    if playing:
        step += 1

        if step >= stop - start:
            step = stop - start - 1
            playing = False

    clock.tick(10 if playing else 30)

pygame.quit()

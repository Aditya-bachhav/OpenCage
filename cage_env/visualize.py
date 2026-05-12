from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
import sys

import pygame

try:
    from cage_env.controller import SignalResponder
    from cage_env.env import OscillationChamberEnv
    from cage_env.session_log import SessionLogger
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from cage_env.controller import SignalResponder
    from cage_env.env import OscillationChamberEnv
    from cage_env.session_log import SessionLogger


WIDTH = 1320
HEIGHT = 780
PANEL_W = 380
WORLD_W = WIDTH - PANEL_W
FPS = 60
BG = (10, 12, 18)
PANEL_BG = (15, 18, 28)
TEXT = (230, 236, 246)
MUTED = (140, 150, 168)
SYSTEM_COLORS = {
    "pendulum": (100, 200, 255),
    "spring": (255, 120, 190),
    "plate": (120, 255, 180),
    "wheel": (255, 205, 95),
    "wave": (190, 150, 255),
}


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    sx = int((x / 10.0) * (WORLD_W - 60)) + 30
    sy = int((1.0 - (y / 10.0)) * (HEIGHT - 70)) + 35
    return sx, sy


def draw_text(surface, font, text, pos, color=TEXT):
    surface.blit(font.render(text, True, color), pos)


def render_glow(surface, pos, radius, color, strength=1.0):
    glow = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
    cx = cy = radius * 3
    for i, alpha in enumerate((26, 18, 12, 7)):
        ring_r = int(radius * (1.0 + i * 0.35) * strength)
        pygame.draw.circle(glow, (*color, alpha), (cx, cy), ring_r)
    surface.blit(glow, (pos[0] - cx, pos[1] - cy), special_flags=pygame.BLEND_PREMULTIPLIED)


def draw_trajectory(surface, history: deque):
    for x, y in history:
        px, py = world_to_screen(x, y)
        pygame.draw.circle(surface, (90, 130, 95), (px, py), 2)


def draw_system(surface, system: dict, small_font):
    x, y = world_to_screen(system["x"], system["y"])
    kind = system["kind"]
    color = SYSTEM_COLORS.get(kind, (160, 160, 160))
    energy = float(system.get("energy", 1.0))
    phase = float(system.get("phase", 0.0))
    radius = max(8, int(system.get("radius", 0.3) * 64))

    pygame.draw.circle(surface, color, (x, y), radius)
    inner = tuple(min(255, int(c + energy * 60)) for c in color)
    pygame.draw.circle(surface, inner, (x, y), max(3, radius - 3))

    tip = (x + int(14 * math.cos(phase)), y + int(14 * math.sin(phase)))
    pygame.draw.line(surface, inner, (x, y), tip, 2)
    render_glow(surface, (x, y), radius, color, strength=0.8 + energy * 0.4)

    label = f"{kind} e={energy:.2f} φ={phase / math.pi:.1f}π"
    surface.blit(small_font.render(label, True, color), (x - 40, y + 16))


def draw_agent(surface, agent: dict, pulse: float):
    x, y = world_to_screen(agent["x"], agent["y"])
    render_glow(surface, (x, y), 14, (70, 255, 120), strength=1.0 + pulse * 0.2)
    pygame.draw.circle(surface, (230, 250, 235), (x, y), 9)
    pygame.draw.circle(surface, (70, 255, 120), (x, y), 6)


def draw_panel(surface, info, font, small_font, mode_label, speed, paused):
    panel_x = WORLD_W
    pygame.draw.rect(surface, PANEL_BG, (panel_x, 0, PANEL_W, HEIGHT))
    pygame.draw.line(surface, (35, 42, 58), (panel_x, 0), (panel_x, HEIGHT), 2)

    signals = info.get("signals", {})
    metrics = info.get("metrics", {})
    draw_text(surface, font, "OSCILLATION CHAMBER", (panel_x + 16, 16), (120, 210, 255))
    draw_text(surface, small_font, mode_label, (panel_x + 16, 42), MUTED)
    draw_text(surface, small_font, f"step {info.get('step', 0)} episode {info.get('episode', 0)}", (panel_x + 16, 60), MUTED)
    draw_text(surface, small_font, f"speed x{speed:.1f} {'paused' if paused else 'running'}", (panel_x + 16, 78), MUTED)

    pos = signals.get("position", (0.0, 0.0))
    draw_text(surface, small_font, f"pos: ({pos[0]:.2f}, {pos[1]:.2f})", (panel_x + 16, 110))
    draw_text(surface, small_font, f"nearest: {signals.get('nearest_system_id', 'none')}", (panel_x + 16, 130))
    draw_text(surface, small_font, f"distance: {signals.get('distance_to_nearest', 0.0):.2f}", (panel_x + 16, 150))
    draw_text(surface, small_font, f"velocity: {signals.get('velocity', (0.0, 0.0))}", (panel_x + 16, 170))

    y = 206
    draw_text(surface, small_font, "attraction", (panel_x + 16, y), (200, 170, 105))
    y += 22
    for system_id, score in sorted(signals.get("attraction_scores", {}).items()):
        draw_text(surface, small_font, f"{system_id:>9}: {score:.3f}", (panel_x + 16, y))
        pygame.draw.rect(surface, SYSTEM_COLORS.get(system_id, (120, 120, 120)), (panel_x + 130, y + 4, int(120 * score), 8))
        y += 18

    y += 8
    draw_text(surface, small_font, "metrics", (panel_x + 16, y), (120, 255, 180))
    y += 22
    for system_id, metric in sorted(metrics.items()):
        draw_text(surface, small_font, f"{system_id:>9}: v={metric.get('visits', 0)} d={metric.get('dwell_time', 0.0):.2f}", (panel_x + 16, y))
        y += 18
        draw_text(surface, small_font, f"          align={metric.get('phase_alignment', 0.0):.2f} sync={metric.get('synchronization_attempts', 0)}", (panel_x + 16, y), MUTED)
        y += 20

    draw_text(surface, small_font, "controls: SPACE pause, R reset, +/- speed, ESC quit", (panel_x + 16, HEIGHT - 40), MUTED)


def load_replay(log_path: Path) -> list[dict]:
    rows = SessionLogger(log_path).read_all()
    return [row for row in rows if row.get("type") == "step"]


def run_live(seed: int):
    env = OscillationChamberEnv()
    controller = SignalResponder(seed=seed)
    obs, info = env.reset(seed=seed)
    return env, controller, obs, info


def main():
    parser = argparse.ArgumentParser(description="Oscillation Chamber visualizer")
    parser.add_argument("--replay", type=Path, help="Replay a JSONL session log")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Oscillation Chamber")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 18)

    replay_frames = load_replay(args.replay) if args.replay else []
    replay_mode = bool(replay_frames)
    frame_index = 0
    paused = False
    speed = 1.0
    live_history = deque(maxlen=256)

    env = controller = obs = info = None
    if not replay_mode:
        env, controller, obs, info = run_live(args.seed)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r and not replay_mode:
                    env, controller, obs, info = run_live(args.seed)
                    live_history.clear()
                    frame_index = 0
                elif event.key in {pygame.K_EQUALS, pygame.K_PLUS}:
                    speed = min(4.0, speed + 0.5)
                elif event.key == pygame.K_MINUS:
                    speed = max(0.25, speed - 0.5)

        if not paused:
            if replay_mode:
                frame_index = min(frame_index + max(1, int(speed)), max(len(replay_frames) - 1, 0))
                info = replay_frames[frame_index]["info"] if replay_frames else info
            else:
                for _ in range(max(1, int(speed))):
                    action = controller.choose_action(info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    live_history.append((env.agent.x, env.agent.y))
                    if terminated or truncated:
                        env, controller, obs, info = run_live(args.seed)
                        live_history.clear()

        frame_info = info
        if frame_info is None:
            continue

        screen.fill(BG)
        pygame.draw.rect(screen, (42, 47, 64), (0, 0, WORLD_W, HEIGHT))

        if replay_mode:
            current_history = deque((step["info"]["agent"]["x"], step["info"]["agent"]["y"]) for step in replay_frames[: frame_index + 1])
        else:
            current_history = live_history

        draw_trajectory(screen, current_history)
        for system in frame_info.get("systems", []):
            draw_system(screen, system, small_font)
        draw_agent(screen, frame_info.get("agent", {"x": 0.0, "y": 0.0}), speed)
        draw_panel(screen, frame_info, font, small_font, "replay" if replay_mode else "live", speed, paused)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

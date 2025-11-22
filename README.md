# Little Prince 3D Scene

This project is an interactive 3D scene inspired by "The Little Prince", built with Python, OpenGL, and Pygame. It features real-time lighting, shadow mapping, animated objects, camera controls, and synchronized music/voiceover.

## Features
- **3D Scene Rendering**: Loads and displays models from the `materials/` folder with textures from `textures/`.
- **Advanced Lighting**: Two real-time lights with shadow mapping for realistic effects.
- **Shooting Star Animation**: Stars named `estrela*` or `stars*` animate with a trailing effect.
- **Camera Controls**:
  - Auto mode: Cycles through saved views, with smooth transitions.
  - Manual mode: Orbit, zoom, and snap/cycle through views.
- **Audio**:
  - Background music (`sound/music.mp3`).
  - Voiceover (`sound/VO-LittlePrince.mp3`) plays at the start of auto mode, lowering music volume and restoring it after.
- **Resource Management**: Cleans up OpenGL and Pygame resources on exit.

## Controls
- `A`: Toggle auto/manual camera mode
- Mouse drag: Orbit camera (manual mode)
- Mouse wheel: Zoom in/out
- Arrow keys: Cycle through saved views (manual mode)
- `[`, `]`: Adjust soft light ambient
- `;`, `'`: Adjust main light ambient
- `ESC`: Quit

## Project Structure
- `main.py`: Main application logic
- `config.py`: Configuration constants
- `model_loader.py`, `texture_loader.py`, `shader.py`: Supporting modules
- `materials/`: Model data files
- `textures/`: Texture images
- `sound/`: Music and voiceover files
- `saved_views.json`: Camera view presets

## Requirements
- Python 3.10+
- Pygame
- PyOpenGL
- numpy
- glm (PyGLM)

Install dependencies with:

    pip install pygame PyOpenGL numpy PyGLM

## Running

    python main.py

## Notes
- Place your own music/voiceover in the `sound/` folder if desired.
- You can adjust camera views in `saved_views.json`.
- All code is robust, well-documented, and cleans up resources on exit.

---

Enjoy exploring the world of The Little Prince in 3D!

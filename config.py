import glm

# Window configuration
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800
WINDOW_TITLE = "Little Prince"

# Camera settings
FOV = 25.0  # Field of view in degrees
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0
CAMERA_POS = glm.vec3(0, 2, 50)
CAMERA_TARGET = glm.vec3(0, 1, 0)
CAMERA_UP = glm.vec3(0, 1, 0)

# Texture unit bindings
TEXTURE_UNITS = {
    "BaseColor": 0,
    "Normal": 1,
    "Roughness": 2,
    "Metallic": 3,
    "Alpha": 4,
    "Emissive": 5
}

import pygame
from pygame.locals import *
from OpenGL.GL import *
import glm
import math
import time
import json
import os

import config
from model_loader import load_model_from_txt
from texture_loader import load_texture
from shader import create_shader_program

CAMERA_FILE = "saved_views.json"
TRANSITION_DURATION = 0.6

SHADOW_WIDTH, SHADOW_HEIGHT = 2048, 2048

class Camera:
    def __init__(self, distance=50.0, yaw=90.0, pitch=0.0, target=glm.vec3(0,0,0)):
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target = target
        self.min_distance = 5.0
        self.max_distance = 150.0

    def copy(self):
        return Camera(self.distance, self.yaw, self.pitch, glm.vec3(self.target))

    def lerp(self, other, t):
        self.yaw = glm.mix(self.yaw, other.yaw, t)
        self.pitch = glm.mix(self.pitch, other.pitch, t)
        self.distance = glm.mix(self.distance, other.distance, t)
        self.target = glm.mix(self.target, other.target, t)

    def process_mouse_drag(self, dx, dy):
        self.yaw += dx * 0.5
        self.pitch -= dy * 0.5
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def zoom(self, zoom_in):
        if zoom_in:
            self.distance = max(self.min_distance, self.distance - 1.0)
        else:
            self.distance = min(self.max_distance, self.distance + 1.0)

    def move_left(self, amount=0.5):
        right = glm.normalize(glm.cross(self.get_front_vector(), glm.vec3(0,1,0)))
        self.target -= right * amount

    def move_right(self, amount=0.5):
        right = glm.normalize(glm.cross(self.get_front_vector(), glm.vec3(0,1,0)))
        self.target += right * amount

    def move_up(self, amount=0.5):
        self.target.y += amount

    def move_down(self, amount=0.5):
        self.target.y -= amount

    def get_front_vector(self):
        yaw_rad = glm.radians(self.yaw)
        pitch_rad = glm.radians(self.pitch)
        front = glm.vec3(
            glm.cos(pitch_rad) * glm.cos(yaw_rad),
            glm.sin(pitch_rad),
            glm.cos(pitch_rad) * glm.sin(yaw_rad)
        )
        return glm.normalize(front)

    def get_position(self):
        yaw_rad = glm.radians(self.yaw)
        pitch_rad = glm.radians(self.pitch)
        x = self.distance * glm.cos(pitch_rad) * glm.cos(yaw_rad)
        y = self.distance * glm.sin(pitch_rad)
        z = self.distance * glm.cos(pitch_rad) * glm.sin(yaw_rad)
        return glm.vec3(x, y, z) + self.target

    def get_view_matrix(self):
        pos = self.get_position()
        up = glm.vec3(0,1,0)
        return glm.lookAt(pos, self.target, up)

    def to_dict(self):
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "distance": self.distance,
            "target": [self.target.x, self.target.y, self.target.z]
        }

    def from_dict(self, data):
        self.yaw = data.get("yaw", self.yaw)
        self.pitch = data.get("pitch", self.pitch)
        self.distance = data.get("distance", self.distance)
        t = data.get("target", [0,0,0])
        self.target = glm.vec3(*t)

def load_views():
    if not os.path.exists(CAMERA_FILE):
        return []
    with open(CAMERA_FILE, "r") as f:
        data = json.load(f)
    views = []
    for v in data:
        cam = Camera()
        cam.from_dict(v)
        views.append(cam)
    return views

def save_views(views):
    data = [v.to_dict() for v in views]
    with open(CAMERA_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(views)} camera views to {CAMERA_FILE}")

def create_depth_shader():
    vertex_shader_source = """
    #version 330 core
    layout(location = 0) in vec3 position;
    uniform mat4 model;
    uniform mat4 lightSpaceMatrix;
    void main()
    {
        gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
    }
    """
    fragment_shader_source = """
    #version 330 core
    void main() {}
    """
    vs = glCreateShader(GL_VERTEX_SHADER)
    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(vs, vertex_shader_source)
    glShaderSource(fs, fragment_shader_source)
    glCompileShader(vs)
    glCompileShader(fs)
    for shader, name in [(vs, "VERTEX"), (fs, "FRAGMENT")]:
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            print(f"ERROR compiling {name} shader: {glGetShaderInfoLog(shader).decode()}")
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        print(f"ERROR linking shader program: {glGetProgramInfoLog(program).decode()}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

def main():
    pygame.init()
    display = (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(config.WINDOW_TITLE)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    shader_program = create_shader_program()
    depth_shader_program = create_depth_shader()

    glUseProgram(shader_program)
    glClearColor(0.1, 0.1, 0.2, 1.0)

    objects = load_model_from_txt("materials", load_texture)

    projection = glm.perspective(glm.radians(config.FOV), display[0]/display[1], config.NEAR_PLANE, config.FAR_PLANE)
    proj_loc = glGetUniformLocation(shader_program, "projection")
    view_loc = glGetUniformLocation(shader_program, "view")
    model_loc = glGetUniformLocation(shader_program, "model")
    light_space_loc = glGetUniformLocation(shader_program, "lightSpaceMatrix")
    light_dir_loc = glGetUniformLocation(shader_program, "lightDir")
    view_pos_loc = glGetUniformLocation(shader_program, "viewPos")
    shadow_map_loc = glGetUniformLocation(shader_program, "shadowMap")

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))

    # Setup shadow map framebuffer
    depth_map_fbo = glGenFramebuffers(1)
    depth_map = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_map)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    border_color = (1.0, 1.0, 1.0, 1.0)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("ERROR::FRAMEBUFFER:: Shadow framebuffer not complete!")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    saved_views = load_views()
    if saved_views:
        current_view_idx = 0
        camera = saved_views[0].copy()
    else:
        current_view_idx = -1
        camera = Camera()

    clock = pygame.time.Clock()
    dragging = False
    last_mouse_pos = (0, 0)
    start_time = time.time()

    plane_object_names = {
        "plane_1", "plane_2", "plane_3", "plane_4", "malakhe"
    }

    axis_weights = [1.0, -1.0, 1.0]
    vertical_bob_amplitude = 0.5
    vertical_bob_speed = 1.0
    sway_amplitude = glm.radians(5)
    sway_speed = 1.0
    sway_axis = glm.normalize(glm.vec3(0, 1, 0))

    propeller_angle = 0.0
    propeller_rotation_speed = 90.0

    # Directional light top-left
    light_pos = glm.vec3(-10, 10, -10)
    light_target = glm.vec3(0, 0, 0)
    light_up = glm.vec3(0, 1, 0)
    light_projection = glm.ortho(-20, 20, -20, 20, 1.0, 40.0)
    light_view = glm.lookAt(light_pos, light_target, light_up)
    light_space_matrix = light_projection * light_view

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        current_time = time.time() - start_time

        propeller_angle += propeller_rotation_speed * dt
        propeller_angle %= 360

        for event in pygame.event.get():
            if event.type == QUIT:
                save_views(saved_views)
                running = False

            elif event.type == KEYDOWN:
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    # manual_move = True  # optional, you can add if you want
                    pass
                if event.key == pygame.K_p:
                    saved_views.append(camera.copy())
                    current_view_idx = len(saved_views) - 1
                    save_views(saved_views)
                    print(f"Saved camera view #{current_view_idx}")

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button in [4, 5]:
                    camera.zoom(event.button == 4)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == MOUSEMOTION and dragging:
                x, y = pygame.mouse.get_pos()
                dx = x - last_mouse_pos[0]
                dy = y - last_mouse_pos[1]
                camera.process_mouse_drag(dx, dy)
                last_mouse_pos = (x, y)

        # 1. Render shadow depth map
        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
        glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glUseProgram(depth_shader_program)

        for obj in objects:
            glUniformMatrix4fv(glGetUniformLocation(depth_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
            glUniformMatrix4fv(glGetUniformLocation(depth_shader_program, "lightSpaceMatrix"), 1, GL_FALSE, glm.value_ptr(light_space_matrix))
            obj.draw(depth_shader_program, config.TEXTURE_UNITS)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # 2. Render scene
        glViewport(0, 0, display[0], display[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader_program)

        view = camera.get_view_matrix()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(light_space_loc, 1, GL_FALSE, glm.value_ptr(light_space_matrix))

        glUniform3fv(light_dir_loc, 1, glm.value_ptr(glm.normalize(light_pos)))
        glUniform3fv(view_pos_loc, 1, glm.value_ptr(camera.get_position()))

        # Bind shadow map texture to texture unit 10
        shadow_tex_unit = 10
        glActiveTexture(GL_TEXTURE0 + shadow_tex_unit)
        glBindTexture(GL_TEXTURE_2D, depth_map)
        glUniform1i(shadow_map_loc, shadow_tex_unit)

        vertical_offset = math.sin(current_time * vertical_bob_speed) * vertical_bob_amplitude
        sway_angle = math.sin(current_time * sway_speed) * sway_amplitude
        vertical_translation = glm.translate(glm.mat4(1.0), glm.vec3(0, vertical_offset, 0))
        sway_rotation = glm.rotate(glm.mat4(1.0), sway_angle, sway_axis)

        rotation_axis = glm.vec3(axis_weights[0], axis_weights[1], axis_weights[2])
        if glm.length(rotation_axis) == 0:
            rotation_axis = glm.vec3(0, 0, 1)
        else:
            rotation_axis = glm.normalize(rotation_axis)

        for obj in objects:
            if obj.name in plane_object_names:
                centroid = obj.centroid
                translate_to_origin = glm.translate(glm.mat4(1.0), -centroid)
                translate_back = glm.translate(glm.mat4(1.0), centroid)
                base_transform = vertical_translation * sway_rotation

                if obj.name == "malakhe":
                    quat = glm.angleAxis(glm.radians(propeller_angle), rotation_axis)
                    rot_mat = glm.mat4_cast(quat)
                    model_matrix = base_transform * translate_back * rot_mat * translate_to_origin
                else:
                    model_matrix = base_transform * translate_back * translate_to_origin
            else:
                model_matrix = glm.mat4(1.0)

            glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matrix))

            color_factor_loc = glGetUniformLocation(shader_program, "colorFactor")
            cf = getattr(obj, "color_factor", (1,1,1,1))
            glUniform4f(color_factor_loc, *cf)

            roughness_loc = glGetUniformLocation(shader_program, "roughness")
            metallic_loc = glGetUniformLocation(shader_program, "metallic")
            glUniform1f(roughness_loc, getattr(obj, "roughness", 1.0))
            glUniform1f(metallic_loc, getattr(obj, "metallic", 0.0))

            obj.draw(shader_program, config.TEXTURE_UNITS)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

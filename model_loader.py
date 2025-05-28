import os
import numpy as np
from OpenGL.GL import *
import ctypes
import glm

class SceneObject:
    def __init__(self, name, vertices, indices, textures, normals):
        self.name = name
        self.vertex_count = len(indices)
        self.textures = textures
        self.vertical_offset = 0.0

        self.raw_vertices = []
        num_vertices = len(vertices) // 5  # x,y,z,u,v layout
        self.centroid = glm.vec3(0,0,0)
        for i in range(num_vertices):
            x, y, z = vertices[i*5], vertices[i*5 + 1], vertices[i*5 + 2]
            pos = glm.vec3(x, y, z)
            self.raw_vertices.append(pos)
            self.centroid += pos
        self.centroid /= num_vertices

        # Combine vertex data: position (3), texCoord (2), normal (3)
        # Normals length should be num_vertices * 3
        vertex_data = []
        for i in range(num_vertices):
            vertex_data.extend(vertices[i*5:i*5+3])      # pos x,y,z
            vertex_data.extend(vertices[i*5+3:i*5+5])    # tex u,v
            vertex_data.extend(normals[i*3:i*3+3])       # normal x,y,z

        vertex_data_np = np.array(vertex_data, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_data_np.nbytes, vertex_data_np, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)

        # position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # texCoord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # normal attribute
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(5 * 4))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def draw(self, shader_program, texture_units):
        if "BaseColor" in self.textures:
            unit = texture_units.get("BaseColor", 0)
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, self.textures["BaseColor"])
            glUniform1i(glGetUniformLocation(shader_program, "texture1"), unit)
            glUniform1i(glGetUniformLocation(shader_program, "useTexture"), True)
        else:
            glUniform1i(glGetUniformLocation(shader_program, "useTexture"), False)

        glUniform4f(glGetUniformLocation(shader_program, "baseColor"), *getattr(self, "base_color", (1,1,1,1)))
        glUniform4f(glGetUniformLocation(shader_program, "colorFactor"), *getattr(self, "color_factor", (1,1,1,1)))
        glUniform1f(glGetUniformLocation(shader_program, "opacity"), getattr(self, "opacity", 1.0))

        # roughness & metallic
        glUniform1f(glGetUniformLocation(shader_program, "roughness"), getattr(self, "roughness", 1.0))
        glUniform1f(glGetUniformLocation(shader_program, "metallic"), getattr(self, "metallic", 0.0))

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.vertex_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

def calculate_flat_normals(vertices, indices):
    """Calculate flat normals per face, assign per vertex"""
    num_vertices = len(vertices) // 5
    normals = np.zeros((num_vertices, 3), dtype=np.float32)

    for i in range(0, len(indices), 3):
        i0 = indices[i]
        i1 = indices[i+1]
        i2 = indices[i+2]

        v0 = np.array(vertices[i0*5:i0*5+3])
        v1 = np.array(vertices[i1*5:i1*5+3])
        v2 = np.array(vertices[i2*5:i2*5+3])

        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm != 0:
            face_normal /= norm

        normals[i0] += face_normal
        normals[i1] += face_normal
        normals[i2] += face_normal

    # Normalize all normals
    for i in range(num_vertices):
        norm = np.linalg.norm(normals[i])
        if norm != 0:
            normals[i] /= norm
        else:
            normals[i] = np.array([0,1,0], dtype=np.float32)  # default up vector

    return normals.flatten().tolist()

def load_model_from_txt(folder_path, texture_loader):
    objects = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder_path, filename), "r") as f:
            lines = f.readlines()
        name = lines[0].split(":")[1].strip()
        base_color_line = lines[1].split(":")[1].strip()
        textures = {}
        base_color_vals = None

        if base_color_line.lower().endswith((".png", ".jpg", ".jpeg")):
            tex_path = os.path.join("textures", base_color_line)
            textures["BaseColor"] = texture_loader(tex_path)
        else:
            try:
                base_color_vals = list(map(float, base_color_line.split()))
            except:
                base_color_vals = [1,1,1,1]

        v_start = lines.index("Vertices:\n") + 1
        i_start = lines.index("Indices:\n")
        vertices = [list(map(float, l.strip().split())) for l in lines[v_start:i_start]]
        indices = [int(i) for l in lines[i_start+1:] for i in l.strip().split()]
        flat_vertices = [coord for v in vertices for coord in v]

        # Calculate or load normals
        normals = calculate_flat_normals(flat_vertices, indices)

        obj = SceneObject(name, flat_vertices, indices, textures, normals)

        if base_color_vals is not None:
            obj.base_color = base_color_vals + [1.0] if len(base_color_vals) == 3 else base_color_vals
        else:
            obj.base_color = (1,1,1,1)

        obj.color_factor = (1,1,1,1)
        obj.opacity = 1.0

        roughness = 1.0
        metallic = 0.0

        for line in lines:
            if line.startswith("Opacity:"):
                try:
                    obj.opacity = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("Roughness:"):
                try:
                    roughness = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("Metallic:"):
                try:
                    metallic = float(line.split(":")[1].strip())
                except:
                    pass

        obj.roughness = roughness
        obj.metallic = metallic

        objects.append(obj)
    return objects

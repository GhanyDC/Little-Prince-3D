from OpenGL.GL import *

vertex_shader = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out vec4 FragPosLightSpace;

void main()
{
    vec4 worldPos = model * vec4(position, 1.0);
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(model))) * normal;

    FragPosLightSpace = lightSpaceMatrix * worldPos;

    gl_Position = projection * view * worldPos;
    TexCoord = texCoord;
}
"""

fragment_shader = """
#version 330 core

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace;

out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D shadowMap;

uniform bool useTexture;
uniform vec4 baseColor;
uniform vec4 colorFactor;

uniform float roughness;
uniform float metallic;

uniform vec3 lightDir;
uniform vec3 viewPos;

// PCF shadow sampling
float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // Check if outside shadow map bounds
    if(projCoords.z > 1.0)
        return 0.0;

    float shadow = 0.0;
    float bias = max(0.003 * (1.0 - dot(normal, lightDir)), 0.001);
    float samples = 4.0;
    float offset = 1.0 / 2048.0; // assuming shadow map size 2048

    for(float x = -1.5; x <= 1.5; x += 1.0)
    {
        for(float y = -1.5; y <= 1.5; y += 1.0)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * offset).r;
            shadow += (projCoords.z - bias) > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= (samples * samples);

    return shadow;
}

void main()
{
    vec4 color = baseColor;
    if(useTexture)
    {
        vec4 texColor = texture(texture1, TexCoord);
        if(texColor.a < 0.1)
            discard;
        color = texColor * colorFactor;
    }

    vec3 norm = normalize(Normal);
    vec3 lightDirection = normalize(-lightDir);

    // Ambient
    vec3 ambient = 0.3 * color.rgb;

    // Diffuse
    float diff = max(dot(norm, lightDirection), 0.0);
    vec3 diffuse = 1.0 * diff * color.rgb;

    // Specular (Blinn-Phong)
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDirection + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 64.0 * (1.0 - roughness));
    vec3 specular = spec * vec3(1.0) * metallic;

    // Shadow
    float shadow = ShadowCalculation(FragPosLightSpace, norm, lightDirection);

    vec3 lighting = ambient + (1.0 - shadow * 0.7) * (diffuse + specular);

    FragColor = vec4(lighting, color.a);
}
"""

def create_shader_program():
    vs = glCreateShader(GL_VERTEX_SHADER)
    fs = glCreateShader(GL_FRAGMENT_SHADER)

    glShaderSource(vs, vertex_shader)
    glShaderSource(fs, fragment_shader)

    glCompileShader(vs)
    glCompileShader(fs)

    for shader, name in [(vs, "VERTEX"), (fs, "FRAGMENT")]:
        success = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not success:
            info_log = glGetShaderInfoLog(shader)
            print(f"ERROR::SHADER_COMPILATION_ERROR of type: {name}\n{info_log.decode()}")

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    success = glGetProgramiv(program, GL_LINK_STATUS)
    if not success:
        info_log = glGetProgramInfoLog(program)
        print(f"ERROR::PROGRAM_LINKING_ERROR\n{info_log.decode()}")

    glDeleteShader(vs)
    glDeleteShader(fs)

    return program

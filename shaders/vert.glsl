#version 330

in vec3 vert;
out vec4 v_color;

uniform vec4 color;
uniform mat4 proj;
uniform mat4 view;

void main() {
    gl_Position = proj * view * vec4(vert, 1.0);
    v_color = color;
}

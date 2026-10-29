#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// Set 0: shared by every object, bound once per frame
layout(set = 0, binding = 0) uniform Camera {
	mat4 view;
	mat4 projection;
} camera;

// Set 1: one descriptor set per object, rebound before each draw
layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	vec4 color;
	vec4 params; // x: 1 to multiply by vertex colors, 0 to use the solid color
} object;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = camera.projection * camera.view * object.model * vec4(in_position, 1.0);

	const vec3 vertex_color = mix(vec3(1.0), in_color, object.params.x);
	out_color = vertex_color * object.color.rgb;
}

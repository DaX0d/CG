#version 450

layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0) uniform Camera {
	mat4 view;
	mat4 projection;
} camera;

layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	vec4 color;
	vec4 params;
} object;

void main() {
	// For a unit sphere the position is also the normal, so scaling it lifts the lines
	// slightly above the surface and they don't flicker against the triangles (z-fighting)
	const vec3 position = in_position * 1.004;

	gl_Position = camera.projection * camera.view * object.model * vec4(position, 1.0);
}

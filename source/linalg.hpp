#pragma once

#include <cmath>
#include <numbers>

// Minimal linear algebra for the labs.
//
// Conventions:
// - right-handed world space, Y is up, the camera looks along -Z in view space;
// - matrices are column-major (m[column][row]), the same memory layout as GLSL mat4;
// - vectors are columns, so transforms compose right to left: P * V * M * v;
// - projection matrices target Vulkan clip space: Y points down, depth is in [0, 1].

namespace linalg {

constexpr float pi = std::numbers::pi_v<float>;

struct Vec3 {
	float x, y, z;
};

struct Vec4 {
	float x, y, z, w;
};

struct Mat4 {
	float m[4][4];
};

static_assert(sizeof(Vec4) == 16 && sizeof(Mat4) == 64, "Must match GLSL vec4 and mat4");

inline float radians(float degrees) {
	return degrees * pi / 180.0f;
}

inline Vec3 operator+(Vec3 a, Vec3 b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline Vec3 operator-(Vec3 a, Vec3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline Vec3 operator*(Vec3 v, float s) {
	return { v.x * s, v.y * s, v.z * s };
}

inline float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(Vec3 v) {
	return std::sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
	const float len = length(v);
	return len > 0.0f ? v * (1.0f / len) : v;
}

inline Mat4 identity() {
	Mat4 r{};
	r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
	return r;
}

// (A * B)[row][col] = sum over k of A[row][k] * B[k][col]
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 4; ++row) {
			for (int k = 0; k < 4; ++k) {
				r.m[col][row] += a.m[k][row] * b.m[col][k];
			}
		}
	}
	return r;
}

// | 1 0 0 tx |
// | 0 1 0 ty |
// | 0 0 1 tz |
// | 0 0 0 1  |
inline Mat4 translate(Vec3 t) {
	Mat4 r = identity();
	r.m[3][0] = t.x;
	r.m[3][1] = t.y;
	r.m[3][2] = t.z;
	return r;
}

inline Mat4 scale(Vec3 s) {
	Mat4 r{};
	r.m[0][0] = s.x;
	r.m[1][1] = s.y;
	r.m[2][2] = s.z;
	r.m[3][3] = 1.0f;
	return r;
}

// | 1  0   0  |
// | 0  c  -s  |
// | 0  s   c  |
inline Mat4 rotateX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[1][1] = c;  r.m[2][1] = -s;
	r.m[1][2] = s;  r.m[2][2] = c;
	return r;
}

// |  c  0  s |
// |  0  1  0 |
// | -s  0  c |
inline Mat4 rotateY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0][0] = c;  r.m[2][0] = s;
	r.m[0][2] = -s; r.m[2][2] = c;
	return r;
}

// | c  -s  0 |
// | s   c  0 |
// | 0   0  1 |
inline Mat4 rotateZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0][0] = c;  r.m[1][0] = -s;
	r.m[0][1] = s;  r.m[1][1] = c;
	return r;
}

// Rotation by angle around an arbitrary axis (Rodrigues' rotation formula)
inline Mat4 rotate(Vec3 axis, float angle) {
	if (length(axis) == 0.0f) {
		return identity();
	}

	const Vec3 a = normalize(axis);
	const float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;

	Mat4 r = identity();
	r.m[0][0] = t * a.x * a.x + c;
	r.m[0][1] = t * a.x * a.y + s * a.z;
	r.m[0][2] = t * a.x * a.z - s * a.y;

	r.m[1][0] = t * a.x * a.y - s * a.z;
	r.m[1][1] = t * a.y * a.y + c;
	r.m[1][2] = t * a.y * a.z + s * a.x;

	r.m[2][0] = t * a.x * a.z + s * a.y;
	r.m[2][1] = t * a.y * a.z - s * a.x;
	r.m[2][2] = t * a.z * a.z + c;
	return r;
}

// View matrix: moves the world so that the camera sits at the origin looking along -Z.
// Rows are the camera basis (right s, up u, back -f), last column is the moved eye position.
inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
	const Vec3 f = normalize(target - eye);
	const Vec3 s = normalize(cross(f, up));
	const Vec3 u = cross(s, f);

	Mat4 r = identity();
	r.m[0][0] = s.x;  r.m[1][0] = s.y;  r.m[2][0] = s.z;  r.m[3][0] = -dot(s, eye);
	r.m[0][1] = u.x;  r.m[1][1] = u.y;  r.m[2][1] = u.z;  r.m[3][1] = -dot(u, eye);
	r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z; r.m[3][2] = dot(f, eye);
	return r;
}

// Perspective projection, f = 1 / tan(fov_y / 2):
// | f/aspect   0        0                 0              |
// | 0         -f        0                 0              |
// | 0          0   far/(near-far)   near*far/(near-far)  |
// | 0          0       -1                 0              |
// z = -near maps to depth 0, z = -far maps to depth 1 (after division by w = -z).
inline Mat4 perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);

	Mat4 r{};
	r.m[0][0] = f / aspect;
	r.m[1][1] = -f;
	r.m[2][2] = z_far / (z_near - z_far);
	r.m[3][2] = z_near * z_far / (z_near - z_far);
	r.m[2][3] = -1.0f;
	return r;
}

// Orthographic projection of the box [left, right] x [bottom, top] x [-near, -far]:
// | 2/(r-l)    0          0          -(r+l)/(r-l) |
// | 0        -2/(t-b)     0           (t+b)/(t-b) |
// | 0          0       -1/(far-near)  -near/(far-near) |
// | 0          0          0            1          |
inline Mat4 orthographic(float left, float right, float bottom, float top, float z_near, float z_far) {
	Mat4 r = identity();
	r.m[0][0] = 2.0f / (right - left);
	r.m[1][1] = -2.0f / (top - bottom);
	r.m[2][2] = -1.0f / (z_far - z_near);
	r.m[3][0] = -(right + left) / (right - left);
	r.m[3][1] = (top + bottom) / (top - bottom);
	r.m[3][2] = -z_near / (z_far - z_near);
	return r;
}

} // namespace linalg

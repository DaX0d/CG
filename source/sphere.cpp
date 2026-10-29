#include "sphere.hpp"

#include <cmath>

namespace geometry {

Mesh generateSphere(uint32_t slices, uint32_t stacks) {
	Mesh mesh;
	mesh.vertices.reserve(sphereVertexCount(slices, stacks));

	// Procedural color from the position in local space: X -> red, Y -> green, Z -> blue
	auto addVertex = [&mesh](linalg::Vec3 p) {
		mesh.vertices.push_back({
			.position = p,
			.color = { 0.5f + 0.5f * p.x, 0.5f + 0.5f * p.y, 0.5f + 0.5f * p.z },
		});
	};

	addVertex({ 0.0f, 1.0f, 0.0f });

	// theta is the polar angle from +Y, phi is the azimuth around Y
	for (uint32_t i = 1; i < stacks; ++i) {
		const float theta = linalg::pi * float(i) / float(stacks);

		for (uint32_t j = 0; j < slices; ++j) {
			const float phi = 2.0f * linalg::pi * float(j) / float(slices);

			addVertex({
				std::sin(theta) * std::cos(phi),
				std::cos(theta),
				std::sin(theta) * std::sin(phi),
			});
		}
	}

	addVertex({ 0.0f, -1.0f, 0.0f });

	const uint32_t north = 0;
	const uint32_t south = uint32_t(mesh.vertices.size()) - 1;

	// Index of the j-th vertex on the i-th ring (i = 1 .. stacks - 1), wraps around in j
	auto ring = [slices](uint32_t i, uint32_t j) {
		return 1 + (i - 1) * slices + j % slices;
	};

	auto& tris = mesh.triangle_indices;

	for (uint32_t j = 0; j < slices; ++j) {
		tris.insert(tris.end(), { north, ring(1, j + 1), ring(1, j) });
	}

	// Each cell between two rings is a quad a-b-d-c split into two triangles:
	// a = (i, j)    b = (i, j + 1)
	// c = (i + 1, j)  d = (i + 1, j + 1)
	for (uint32_t i = 1; i + 1 < stacks; ++i) {
		for (uint32_t j = 0; j < slices; ++j) {
			const uint32_t a = ring(i, j), b = ring(i, j + 1);
			const uint32_t c = ring(i + 1, j), d = ring(i + 1, j + 1);

			tris.insert(tris.end(), { a, b, d, a, d, c });
		}
	}

	for (uint32_t j = 0; j < slices; ++j) {
		tris.insert(tris.end(), { ring(stacks - 1, j), ring(stacks - 1, j + 1), south });
	}

	auto& lines = mesh.line_indices;

	for (uint32_t j = 0; j < slices; ++j) {
		lines.insert(lines.end(), { north, ring(1, j) });
		for (uint32_t i = 1; i + 1 < stacks; ++i) {
			lines.insert(lines.end(), { ring(i, j), ring(i + 1, j) });
		}
		lines.insert(lines.end(), { ring(stacks - 1, j), south });
	}

	for (uint32_t i = 1; i < stacks; ++i) {
		for (uint32_t j = 0; j < slices; ++j) {
			lines.insert(lines.end(), { ring(i, j), ring(i, j + 1) });
		}
	}

	return mesh;
}

} // namespace geometry

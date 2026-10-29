#pragma once

#include <cstdint>
#include <vector>

#include "linalg.hpp"

namespace geometry {

struct Vertex {
	linalg::Vec3 position;
	linalg::Vec3 color;
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> triangle_indices; // triangle list, counter-clockwise when seen from outside
	std::vector<uint32_t> line_indices;     // line list: parallels and meridians, for wireframe
};

constexpr uint32_t sphereVertexCount(uint32_t slices, uint32_t stacks) {
	return 2 + (stacks - 1) * slices;
}

// UV sphere of radius 1 centered at the origin.
// slices: segments around the Y axis (>= 3), stacks: segments from pole to pole (>= 2).
// Two poles plus (stacks - 1) rings of 'slices' vertices, e.g. 12 x 9 gives 98 vertices.
Mesh generateSphere(uint32_t slices, uint32_t stacks);

} // namespace geometry

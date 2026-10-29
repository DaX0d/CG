#include "application.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include "graphics.hpp"
#include "linalg.hpp"
#include "sphere.hpp"

namespace application {

namespace {

using linalg::Mat4;
using linalg::Vec3;
using linalg::Vec4;

constexpr uint32_t max_objects = 16;

enum class Projection : int {
	perspective,
	orthographic,
};

enum class Trajectory : int {
	none,
	circle,
	lissajous,
	trefoil,
	rose,
};

constexpr const char* trajectory_names[] = {
	"None",
	"Circle",
	"Lissajous curve",
	"Trefoil knot",
	"Rose curve",
};

constexpr const char* trajectory_formulas[] = {
	"",
	"x = R cos t,  z = R sin t",
	"x = R cos(a t),  y = H sin(b t),  z = R sin(c t)",
	"x = R/3 (sin t + 2 sin 2t),  y = -H sin 3t,  z = R/3 (cos t - 2 cos 2t)",
	"r = R cos(k t),  x = r cos t,  y = H sin(k t),  z = r sin t",
};

// Must match uniform blocks in shaders/*.vert (std140 layout)
struct CameraUniforms {
	Mat4 view;
	Mat4 projection;
};

struct ObjectUniforms {
	Mat4 model;
	Vec4 color;
	Vec4 params; // x: 1 to multiply by vertex colors, 0 to use the solid color
};

struct Camera {
	Projection projection = Projection::perspective;
	float fov = 60.0f;          // vertical, degrees
	float ortho_height = 12.0f; // visible height in world units
	float z_near = 0.1f;
	float z_far = 100.0f;

	// The camera orbits around the origin and always looks at it
	float distance = 12.0f;
	float yaw = 20.0f;   // degrees
	float pitch = 25.0f; // degrees
};

struct Animation {
	bool playing = true;
	float speed = 1.0f;
	float time = 0.0f; // advances by speed * dt while playing, drives both motion and spin

	Trajectory trajectory = Trajectory::none;
	float radius = 3.0f;
	float height = 1.0f;
	Vec3 frequency = { 1.0f, 3.0f, 2.0f }; // Lissajous a, b, c
	int petals = 3;                        // Rose k

	float spin_speed = 45.0f; // degrees per second around spin_axis
	Vec3 spin_axis = { 0.0f, 1.0f, 0.0f };
};

struct Object {
	std::string name;

	Vec3 position = { 0.0f, 0.0f, 0.0f };
	Vec3 rotation = { 0.0f, 0.0f, 0.0f }; // Euler angles in degrees, applied in X, Y, Z order
	Vec3 scale = { 1.0f, 1.0f, 1.0f };

	Vec3 color = { 1.0f, 1.0f, 1.0f };
	bool vertex_colors = true;

	Animation animation;

	ObjectUniforms uniforms{};

	// Every object owns a uniform buffer and a descriptor set that points to it
	graphics::Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

VkDescriptorSetLayout camera_set_layout;
VkDescriptorSetLayout object_set_layout;
VkPipelineLayout pipeline_layout;
VkPipeline mesh_pipeline;
VkPipeline wireframe_pipeline;
VkDescriptorPool descriptor_pool;

Camera camera;
CameraUniforms camera_uniforms;
graphics::Buffer camera_uniform_buffer;
VkDescriptorSet camera_descriptor_set;

// One sphere mesh is shared by all objects
int sphere_slices = 12;
int sphere_stacks = 9;
graphics::Buffer vertex_buffer;
graphics::Buffer triangle_index_buffer;
graphics::Buffer line_index_buffer;
uint32_t triangle_index_count;
uint32_t line_index_count;

std::vector<Object> objects;
int selected_object = 0;
int next_object_number = 1;
bool all_paused = false;

bool show_wireframe = false;
Vec3 wireframe_color = { 0.05f, 0.05f, 0.05f };
Vec3 background_color = { 0.08f, 0.08f, 0.1f };

double previous_time = -1.0;

VkDescriptorSetLayout createUniformSetLayout() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	if (vkCreateDescriptorSetLayout(graphics::internal::context.device, &layout_info,
	                                nullptr, &layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout\n";
	}

	return layout;
}

VkDescriptorSet allocateUniformSet(VkDescriptorSetLayout layout, const graphics::Buffer& buffer) {
	auto& context = graphics::internal::context;

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &layout,
	};

	VkDescriptorSet set;
	if (vkAllocateDescriptorSets(context.device, &allocate_info, &set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set\n";
		return VK_NULL_HANDLE;
	}

	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = buffer.size,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

	return set;
}

Object newObject() {
	Object object;
	object.name = "Sphere " + std::to_string(next_object_number);

	const ImColor color = ImColor::HSV(std::fmod(0.13f * float(next_object_number), 1.0f), 0.45f, 1.0f);
	object.color = { color.Value.x, color.Value.y, color.Value.z };

	++next_object_number;
	return object;
}

bool addObject(Object object) {
	if (objects.size() >= max_objects) {
		return false;
	}

	if (!graphics::createBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                            object.uniform_buffer)) {
		return false;
	}

	object.descriptor_set = allocateUniformSet(object_set_layout, object.uniform_buffer);
	if (object.descriptor_set == VK_NULL_HANDLE) {
		graphics::destroyBuffer(object.uniform_buffer);
		return false;
	}

	objects.push_back(std::move(object));
	return true;
}

// The GPU must not be using the object's resources, wait for the queue before calling this
void destroyObject(Object& object) {
	if (object.descriptor_set != VK_NULL_HANDLE) {
		vkFreeDescriptorSets(graphics::internal::context.device, descriptor_pool,
		                     1, &object.descriptor_set);
		object.descriptor_set = VK_NULL_HANDLE;
	}

	graphics::destroyBuffer(object.uniform_buffer);
}

template <typename T>
bool createFilledBuffer(const std::vector<T>& data, VkBufferUsageFlags usage, graphics::Buffer& buffer) {
	const VkDeviceSize size = data.size() * sizeof(T);
	return graphics::createBuffer(size, usage, buffer) &&
	       graphics::uploadBuffer(buffer, data.data(), size);
}

bool createSphere() {
	const geometry::Mesh mesh = geometry::generateSphere(uint32_t(sphere_slices),
	                                                     uint32_t(sphere_stacks));

	triangle_index_count = uint32_t(mesh.triangle_indices.size());
	line_index_count = uint32_t(mesh.line_indices.size());

	return createFilledBuffer(mesh.vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) &&
	       createFilledBuffer(mesh.triangle_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, triangle_index_buffer) &&
	       createFilledBuffer(mesh.line_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, line_index_buffer);
}

void destroySphere() {
	graphics::destroyBuffer(vertex_buffer);
	graphics::destroyBuffer(triangle_index_buffer);
	graphics::destroyBuffer(line_index_buffer);
}

bool createResources() {
	auto& context = graphics::internal::context;

	camera_set_layout = createUniformSetLayout();
	object_set_layout = createUniformSetLayout();
	if (camera_set_layout == VK_NULL_HANDLE || object_set_layout == VK_NULL_HANDLE) {
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = {
		camera_set_layout, // set = 0
		object_set_layout, // set = 1
	};

	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(Vec4), // wireframe color
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(geometry::Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(geometry::Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(geometry::Vertex, color),
		},
	};

	mesh_pipeline = graphics::createGraphicsPipeline({
		.vertex_shader = "shaders/mesh.vert.spv",
		.fragment_shader = "shaders/mesh.frag.spv",
		.layout = pipeline_layout,
		.vertex_bindings = std::span(&vertex_binding, 1),
		.vertex_attributes = vertex_attributes,
	});

	// Wireframe shaders only read positions
	wireframe_pipeline = graphics::createGraphicsPipeline({
		.vertex_shader = "shaders/wireframe.vert.spv",
		.fragment_shader = "shaders/wireframe.frag.spv",
		.layout = pipeline_layout,
		.vertex_bindings = std::span(&vertex_binding, 1),
		.vertex_attributes = std::span(vertex_attributes, 1),
		.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		.cull_mode = VK_CULL_MODE_NONE,
		.depth_write = false,
	});

	if (mesh_pipeline == VK_NULL_HANDLE || wireframe_pipeline == VK_NULL_HANDLE) {
		return false;
	}

	// One set for the camera plus one set per object
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1 + max_objects,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = 1 + max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	if (!graphics::createBuffer(sizeof(CameraUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                            camera_uniform_buffer)) {
		return false;
	}

	camera_descriptor_set = allocateUniformSet(camera_set_layout, camera_uniform_buffer);
	if (camera_descriptor_set == VK_NULL_HANDLE) {
		return false;
	}

	if (!createSphere()) {
		return false;
	}

	Object center = newObject();
	center.color = { 1.0f, 1.0f, 1.0f };
	center.animation.spin_speed = 30.0f;

	Object lissajous = newObject();
	lissajous.scale = { 0.5f, 0.5f, 0.5f };
	lissajous.color = { 1.0f, 0.85f, 0.5f };
	lissajous.animation.trajectory = Trajectory::lissajous;
	lissajous.animation.radius = 4.0f;
	lissajous.animation.height = 2.0f;
	lissajous.animation.speed = 0.6f;
	lissajous.animation.spin_speed = 90.0f;
	lissajous.animation.spin_axis = { 1.0f, 0.0f, 0.0f };

	Object trefoil = newObject();
	trefoil.scale = { 0.35f, 0.35f, 0.35f };
	trefoil.color = { 0.6f, 0.85f, 1.0f };
	trefoil.animation.trajectory = Trajectory::trefoil;
	trefoil.animation.radius = 4.5f;
	trefoil.animation.height = 1.0f;
	trefoil.animation.speed = 0.4f;
	trefoil.animation.spin_speed = 120.0f;
	trefoil.animation.spin_axis = { 0.0f, 0.0f, 1.0f };

	return addObject(std::move(center)) &&
	       addObject(std::move(lissajous)) &&
	       addObject(std::move(trefoil));
}

// Position on the trajectory at parameter t
Vec3 trajectoryPoint(const Animation& a, float t) {
	using std::cos, std::sin;

	switch (a.trajectory) {
	case Trajectory::circle:
		return { a.radius * cos(t), 0.0f, a.radius * sin(t) };

	case Trajectory::lissajous:
		return {
			a.radius * cos(a.frequency.x * t),
			a.height * sin(a.frequency.y * t),
			a.radius * sin(a.frequency.z * t),
		};

	case Trajectory::trefoil: {
		// The planar part of the knot stays within distance 3 from the center
		const float k = a.radius / 3.0f;
		return {
			k * (sin(t) + 2.0f * sin(2.0f * t)),
			-a.height * sin(3.0f * t),
			k * (cos(t) - 2.0f * cos(2.0f * t)),
		};
	}

	case Trajectory::rose: {
		// Height is at its extreme exactly when r = 0, so the path goes over or under the center
		const float k = float(a.petals);
		const float r = a.radius * cos(k * t);
		return { r * cos(t), a.height * sin(k * t), r * sin(t) };
	}

	case Trajectory::none:
		break;
	}

	return { 0.0f, 0.0f, 0.0f };
}

Mat4 modelMatrix(const Object& object) {
	const Animation& a = object.animation;

	const Vec3 position = object.position + trajectoryPoint(a, a.time);
	const float spin = linalg::radians(a.spin_speed * a.time);

	// Applied right to left: scale, rotation from the UI, animated spin, translation
	return linalg::translate(position) *
	       linalg::rotate(a.spin_axis, spin) *
	       linalg::rotateZ(linalg::radians(object.rotation.z)) *
	       linalg::rotateY(linalg::radians(object.rotation.y)) *
	       linalg::rotateX(linalg::radians(object.rotation.x)) *
	       linalg::scale(object.scale);
}

void updateCamera() {
	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;

	const float yaw = linalg::radians(camera.yaw);
	const float pitch = linalg::radians(camera.pitch);

	const Vec3 eye = Vec3{
		std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		std::cos(pitch) * std::cos(yaw),
	} * camera.distance;

	camera_uniforms.view = linalg::lookAt(eye, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

	const float z_far = std::max(camera.z_far, camera.z_near + 0.01f);

	if (camera.projection == Projection::perspective) {
		camera_uniforms.projection = linalg::perspective(linalg::radians(camera.fov), aspect,
		                                                 camera.z_near, z_far);
	} else {
		const float half_height = camera.ortho_height * 0.5f;
		const float half_width = half_height * aspect;
		camera_uniforms.projection = linalg::orthographic(-half_width, half_width,
		                                                  -half_height, half_height,
		                                                  camera.z_near, z_far);
	}
}

void drawObjectEditor(Object& object) {
	Animation& animation = object.animation;

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", &object.position.x, 0.05f, 0.0f, 0.0f, "%.2f");
	ImGui::DragFloat3("Rotation", &object.rotation.x, 1.0f, -360.0f, 360.0f, "%.0f deg");
	ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.05f, 10.0f, "%.2f",
	                  ImGuiSliderFlags_AlwaysClamp);
	if (ImGui::Button("Reset transform")) {
		object.position = { 0.0f, 0.0f, 0.0f };
		object.rotation = { 0.0f, 0.0f, 0.0f };
		object.scale = { 1.0f, 1.0f, 1.0f };
	}

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Color", &object.color.x);
	ImGui::Checkbox("Multiply by vertex colors", &object.vertex_colors);

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(animation.playing ? "Pause###play" : "Play###play", ImVec2(80.0f, 0.0f))) {
		animation.playing = !animation.playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		animation.time = 0.0f;
	}
	ImGui::SameLine();
	ImGui::Text("t = %.2f", animation.time);

	ImGui::SliderFloat("Speed", &animation.speed, 0.0f, 5.0f, "%.2fx");
	ImGui::SliderFloat("Spin speed", &animation.spin_speed, -360.0f, 360.0f, "%.0f deg/s");
	ImGui::DragFloat3("Spin axis", &animation.spin_axis.x, 0.01f, -1.0f, 1.0f, "%.2f");

	int trajectory = int(animation.trajectory);
	ImGui::Combo("Trajectory", &trajectory, trajectory_names, IM_ARRAYSIZE(trajectory_names));
	animation.trajectory = Trajectory(trajectory);

	if (animation.trajectory == Trajectory::none) {
		return;
	}

	ImGui::TextWrapped("%s", trajectory_formulas[trajectory]);

	ImGui::SliderFloat("Radius (R)", &animation.radius, 0.5f, 10.0f, "%.2f");
	if (animation.trajectory != Trajectory::circle) {
		ImGui::SliderFloat("Height (H)", &animation.height, 0.0f, 5.0f, "%.2f");
	}
	if (animation.trajectory == Trajectory::lissajous) {
		ImGui::DragFloat3("Frequencies (a, b, c)", &animation.frequency.x, 0.01f, 0.0f, 8.0f, "%.2f");
	}
	if (animation.trajectory == Trajectory::rose) {
		ImGui::SliderInt("Petals (k)", &animation.petals, 1, 12, "%d", ImGuiSliderFlags_AlwaysClamp);
	}
}

void drawInterface() {
	auto& context = graphics::internal::context;

	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 680.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 1: Sphere");

	const ImGuiIO& io = ImGui::GetIO();
	ImGui::Text("%.0f FPS (%.2f ms)", io.Framerate, 1000.0f / io.Framerate);

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		int projection = int(camera.projection);
		ImGui::RadioButton("Perspective", &projection, int(Projection::perspective));
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &projection, int(Projection::orthographic));
		camera.projection = Projection(projection);

		if (camera.projection == Projection::perspective) {
			ImGui::SliderFloat("Field of view", &camera.fov, 10.0f, 150.0f, "%.0f deg");
		} else {
			ImGui::SliderFloat("View height", &camera.ortho_height, 1.0f, 50.0f, "%.1f");
		}

		ImGui::SliderFloat("Near plane", &camera.z_near, 0.01f, 10.0f, "%.2f",
		                   ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Far plane", &camera.z_far, 1.0f, 500.0f, "%.1f",
		                   ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
	}

	if (ImGui::CollapsingHeader("Camera")) {
		ImGui::SliderFloat("Distance", &camera.distance, 1.0f, 50.0f, "%.1f");
		ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg",
		                   ImGuiSliderFlags_AlwaysClamp);
		ImGui::ColorEdit3("Background", &background_color.x);
	}

	if (ImGui::CollapsingHeader("Sphere mesh")) {
		bool changed = ImGui::SliderInt("Slices", &sphere_slices, 3, 64, "%d",
		                                ImGuiSliderFlags_AlwaysClamp);
		changed |= ImGui::SliderInt("Stacks", &sphere_stacks, 2, 48, "%d",
		                            ImGuiSliderFlags_AlwaysClamp);

		if (changed) {
			vkQueueWaitIdle(context.graphics_queue);
			destroySphere();
			createSphere();
		}

		ImGui::Text("Vertices: %u, triangles: %u",
		            geometry::sphereVertexCount(uint32_t(sphere_slices), uint32_t(sphere_stacks)),
		            triangle_index_count / 3);

		ImGui::Checkbox("Wireframe", &show_wireframe);
		ImGui::SameLine();
		ImGui::ColorEdit3("Line color", &wireframe_color.x, ImGuiColorEditFlags_NoInputs);
	}

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Pause all animations", &all_paused);

		const ImVec2 list_size(-FLT_MIN, 5.0f * ImGui::GetTextLineHeightWithSpacing());
		if (ImGui::BeginListBox("##objects", list_size)) {
			for (int i = 0; i < int(objects.size()); ++i) {
				ImGui::PushID(i);
				if (ImGui::Selectable(objects[i].name.c_str(), i == selected_object)) {
					selected_object = i;
				}
				ImGui::PopID();
			}
			ImGui::EndListBox();
		}

		ImGui::BeginDisabled(objects.size() >= max_objects);
		if (ImGui::Button("Add")) {
			Object object = newObject();
			object.scale = { 0.4f, 0.4f, 0.4f };
			object.animation.trajectory = Trajectory::circle;
			object.animation.radius = 2.0f + 0.5f * float(objects.size());

			if (addObject(std::move(object))) {
				selected_object = int(objects.size()) - 1;
			}
		}
		ImGui::EndDisabled();

		ImGui::SameLine();

		ImGui::BeginDisabled(objects.size() <= 1);
		if (ImGui::Button("Remove selected")) {
			vkQueueWaitIdle(context.graphics_queue);
			destroyObject(objects[selected_object]);
			objects.erase(objects.begin() + selected_object);
			selected_object = std::min(selected_object, int(objects.size()) - 1);
		}
		ImGui::EndDisabled();

		ImGui::SameLine();
		ImGui::TextDisabled("%d / %d", int(objects.size()), int(max_objects));
	}

	if (selected_object >= 0 && selected_object < int(objects.size())) {
		Object& object = objects[selected_object];

		const std::string header = "Selected: " + object.name + "###selected";
		if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			drawObjectEditor(object);
		}
	}

	ImGui::End();
}

} // namespace

bool initialize() {
	if (!createResources()) {
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	for (Object& object : objects) {
		destroyObject(object);
	}
	objects.clear();

	graphics::destroyBuffer(camera_uniform_buffer);
	destroySphere();

	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyPipeline(context.device, wireframe_pipeline, nullptr);
	vkDestroyPipeline(context.device, mesh_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, object_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, camera_set_layout, nullptr);
}

void update(double time) {
	// Clamp the step so the animation doesn't jump after the window was blocked (e.g. dragged)
	const float dt = previous_time < 0.0 ? 0.0f : float(std::min(time - previous_time, 0.1));
	previous_time = time;

	drawInterface();

	for (Object& object : objects) {
		Animation& animation = object.animation;
		if (animation.playing && !all_paused) {
			animation.time += animation.speed * dt;
		}

		object.uniforms = {
			.model = modelMatrix(object),
			.color = { object.color.x, object.color.y, object.color.z, 1.0f },
			.params = { object.vertex_colors ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f },
		};
	}

	updateCamera();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	const VkCommandBuffer cmd = fd.command_buffer;

	if (cmd == VK_NULL_HANDLE) {
		return;
	}

	// prepare() has waited for the previous frame to finish on the GPU,
	// so nothing reads the uniform buffers now and they can be overwritten
	graphics::uploadBuffer(camera_uniform_buffer, &camera_uniforms, sizeof(camera_uniforms));
	for (const Object& object : objects) {
		graphics::uploadBuffer(object.uniform_buffer, &object.uniforms, sizeof(object.uniforms));
	}

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin_info);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { background_color.x, background_color.y, background_color.z, 1.0f } } },
		{ .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	if (vertex_buffer.buffer != VK_NULL_HANDLE) {
		const VkViewport viewport = {
			.width = float(context.swapchain_extent.width),
			.height = float(context.swapchain_extent.height),
			.maxDepth = 1.0f,
		};

		const VkRect2D scissor = { .extent = context.swapchain_extent };

		vkCmdSetViewport(cmd, 0, 1, &viewport);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);

		// Set 0 (camera) is bound once and stays bound for both pipelines since they share
		// the layout. Set 1 is switched before each draw: the same mesh gets different uniforms.
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &camera_descriptor_set, 0, nullptr);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline);
		vkCmdBindIndexBuffer(cmd, triangle_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		for (const Object& object : objects) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
			                        1, 1, &object.descriptor_set, 0, nullptr);
			vkCmdDrawIndexed(cmd, triangle_index_count, 1, 0, 0, 0);
		}

		if (show_wireframe) {
			const Vec4 line_color = { wireframe_color.x, wireframe_color.y, wireframe_color.z, 1.0f };

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
			vkCmdBindIndexBuffer(cmd, line_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
			                   0, sizeof(line_color), &line_color);

			for (const Object& object : objects) {
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				                        1, 1, &object.descriptor_set, 0, nullptr);
				vkCmdDrawIndexed(cmd, line_index_count, 1, 0, 0, 0);
			}
		}
	}

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

} // namespace application

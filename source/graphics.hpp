#pragma once

#include <span>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

namespace graphics {

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkDeviceSize size = 0;
};

// Buffer memory is host-visible, so its contents can be written directly with uploadBuffer
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer);
void destroyBuffer(Buffer& buffer);
bool uploadBuffer(const Buffer& buffer, const void* data, VkDeviceSize size);

// Loads SPIR-V file, path is relative to the working directory (project root)
VkShaderModule loadShaderModule(const char* path);

struct GraphicsPipelineInfo {
	const char* vertex_shader;
	const char* fragment_shader;
	VkPipelineLayout layout;
	std::span<const VkVertexInputBindingDescription> vertex_bindings;
	std::span<const VkVertexInputAttributeDescription> vertex_attributes;
	VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
	bool depth_write = true;
};

// Pipeline for the main render pass with dynamic viewport and scissor, so it survives window resizing
VkPipeline createGraphicsPipeline(const GraphicsPipelineInfo& info);

} // namespace graphics

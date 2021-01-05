#pragma once

#include <vulkan/vulkan.hpp>
#include "Device.h"

namespace Image {
    void Create(Device* device, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory);
    void TransitionLayout(Device* device, vk::CommandPool commandPool, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    vk::ImageView CreateView(Device* device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags);
    void CopyFromBuffer(Device* device, vk::CommandPool commandPool, vk::Buffer buffer, vk::Image& image, uint32_t width, uint32_t height);
    void FromFile(Device* device, vk::CommandPool commandPool, const char* path, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageLayout layout, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory);
}

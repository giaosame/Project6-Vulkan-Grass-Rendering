#pragma once

#include <vulkan/vulkan.h>
#include "Device.h"

namespace BufferUtils {
    void CreateBuffer(Device* device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory);
    void CopyBuffer(Device* device, vk::CommandPool commandPool, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    void CreateBufferFromData(Device* device, vk::CommandPool commandPool, void* bufferData, vk::DeviceSize bufferSize, vk::BufferUsageFlags bufferUsage, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory);
}

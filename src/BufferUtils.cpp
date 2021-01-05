#include "BufferUtils.h"
#include "Instance.h"

void BufferUtils::CreateBuffer(Device* device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    // Create buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(usage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    try {
        buffer = device->GetLogicalDevice().createBuffer(bufferInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create vertex buffer");
    }

    // Query buffer's memory requirements
    vk::MemoryRequirements memRequirements = device->GetLogicalDevice().getBufferMemoryRequirements(buffer);

    // Allocate memory in device
    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(device->GetInstance()->GetMemoryTypeIndex(memRequirements.memoryTypeBits, properties));

    try {
        bufferMemory = device->GetLogicalDevice().allocateMemory(allocInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate vertex buffer");
    }

    // Associate allocated memory with vertex buffer
    device->GetLogicalDevice().bindBufferMemory(buffer, bufferMemory, 0);
}

void BufferUtils::CopyBuffer(Device* device, vk::CommandPool commandPool, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandPool(commandPool);
    allocInfo.setCommandBufferCount(1);

    vk::CommandBuffer commandBuffer;
    device->GetLogicalDevice().allocateCommandBuffers(&allocInfo, &commandBuffer);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion;
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    commandBuffer.end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&commandBuffer);

    device->GetQueue(QueueFlags::Graphics).submit(submitInfo, nullptr);
    device->GetQueue(QueueFlags::Graphics).waitIdle();
    device->GetLogicalDevice().freeCommandBuffers(commandPool, 1, &commandBuffer);
}

void BufferUtils::CreateBufferFromData(Device* device, vk::CommandPool commandPool, void* bufferData, vk::DeviceSize bufferSize, vk::BufferUsageFlags bufferUsage, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    // Create the staging buffer
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    vk::BufferUsageFlags stagingUsage(vk::BufferUsageFlagBits::eTransferSrc);
    vk::MemoryPropertyFlags stagingProperties(vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    BufferUtils::CreateBuffer(device, bufferSize, stagingUsage, stagingProperties, stagingBuffer, stagingBufferMemory);

    // Fill the staging buffer
    void* data = device->GetLogicalDevice().mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(data, bufferData, static_cast<size_t>(bufferSize));
    device->GetLogicalDevice().unmapMemory(stagingBufferMemory);

    // Create the buffer
    vk::BufferUsageFlags usage = vk::BufferUsageFlags(vk::BufferUsageFlagBits::eTransferDst) | bufferUsage;
    vk::MemoryPropertyFlags flags(vk::MemoryPropertyFlagBits::eDeviceLocal);
    BufferUtils::CreateBuffer(device, bufferSize, usage, flags, buffer, bufferMemory);

    // Copy data from staging to buffer
    BufferUtils::CopyBuffer(device, commandPool, stagingBuffer, buffer, bufferSize);

    // No need for the staging buffer anymore
    device->GetLogicalDevice().destroyBuffer(stagingBuffer);
    device->GetLogicalDevice().freeMemory(stagingBufferMemory);
}

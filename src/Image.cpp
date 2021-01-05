#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Image.h"
#include "Device.h"
#include "Instance.h"
#include "BufferUtils.h"

void Image::Create(Device* device, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
    // Create Vulkan image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent(vk::Extent3D(width, height, 1));
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setTiling(tiling);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    imageInfo.setUsage(usage);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);

    try {
        image = device->GetLogicalDevice().createImage(imageInfo);
    } 
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create image");
    }

    // Allocate memory for the image
    vk::MemoryRequirements memRequirements = device->GetLogicalDevice().getImageMemoryRequirements(image);
    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(device->GetInstance()->GetMemoryTypeIndex(memRequirements.memoryTypeBits, properties));
   
    try {
        imageMemory = device->GetLogicalDevice().allocateMemory(allocInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate image memory");
    }

    // Bind the image
    device->GetLogicalDevice().bindImageMemory(image, imageMemory, 0);
}

void Image::TransitionLayout(Device* device, vk::CommandPool commandPool, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    auto hasStencilComponent = [](vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    };

    // Use an image memory barrier (type of pipeline barrier) to transition image layout
    vk::ImageMemoryBarrier barrier;
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
  
    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    
        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }
    }
    else {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    }
  
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
  
    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;
  
    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits(0));
        barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
        barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    
        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits(0));
        barrier.setDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                 vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
    } else {
        throw std::invalid_argument("Unsupported layout transition");
    }

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandPool(commandPool);
    allocInfo.setCommandBufferCount(1);

    vk::CommandBuffer commandBuffer;
    device->GetLogicalDevice().allocateCommandBuffers(&allocInfo, &commandBuffer);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    commandBuffer.begin(beginInfo);
    
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);
  
    commandBuffer.end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&commandBuffer);

    device->GetQueue(QueueFlags::Graphics).submit(submitInfo, nullptr);
    device->GetQueue(QueueFlags::Graphics).waitIdle();
    device->GetLogicalDevice().freeCommandBuffers(commandPool, 1, &commandBuffer);
}

vk::ImageView Image::CreateView(Device* device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.setImage(image);
    viewInfo.setViewType(vk::ImageViewType::e2D);
    viewInfo.setFormat(format);
    
    // Describe the image's purpose and which part of the image should be accessed
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vk::ImageView imageView;
    try {
        imageView = device->GetLogicalDevice().createImageView(viewInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to texture image view");
    }

    return imageView;
}

void Image::CopyFromBuffer(Device* device, vk::CommandPool commandPool, vk::Buffer buffer, vk::Image& image, uint32_t width, uint32_t height) {
    // Specify which part of the buffer is going to be copied to which part of the image
    vk::BufferImageCopy region;
    region.setBufferOffset(0);
    region.setBufferRowLength(0);
    region.setBufferImageHeight(0);

    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.setImageOffset(vk::Offset3D{ 0, 0, 0 });
    region.setImageExtent(vk::Extent3D{ width, height, 1 });

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandPool(commandPool);
    allocInfo.setCommandBufferCount(1);

    vk::CommandBuffer commandBuffer;
    device->GetLogicalDevice().allocateCommandBuffers(&allocInfo, &commandBuffer);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    commandBuffer.begin(beginInfo);
   
    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    commandBuffer.end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&commandBuffer);

    device->GetQueue(QueueFlags::Transfer).submit(submitInfo, nullptr);
    device->GetQueue(QueueFlags::Transfer).waitIdle();
    device->GetLogicalDevice().freeCommandBuffers(commandPool, 1, &commandBuffer);
}

void Image::FromFile(Device* device, vk::CommandPool commandPool, const char* path, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageLayout layout, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("Failed to load texture image");
    }

    // Create staging buffer
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    vk::BufferUsageFlags stagingUsage(vk::BufferUsageFlagBits::eTransferSrc);
    vk::MemoryPropertyFlags stagingProperties(vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    BufferUtils::CreateBuffer(device, imageSize, stagingUsage, stagingProperties, stagingBuffer, stagingBufferMemory);

    // Copy pixel values to the buffer
    void* data = device->GetLogicalDevice().mapMemory(stagingBufferMemory, 0, imageSize);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    device->GetLogicalDevice().unmapMemory(stagingBufferMemory);

    // Free pixel array
    stbi_image_free(pixels);

    // Create Vulkan image
    Image::Create(device, texWidth, texHeight, format, tiling, vk::ImageUsageFlags(vk::ImageUsageFlagBits::eTransferDst) | usage, properties, image, imageMemory);

    // Copy the staging buffer to the texture image
    // --> First need to transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    Image::TransitionLayout(device, commandPool, image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    Image::CopyFromBuffer(device, commandPool, stagingBuffer, image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    // Transition texture image for shader access
    Image::TransitionLayout(device, commandPool, image, format, vk::ImageLayout::eTransferDstOptimal, layout);
     
    // No need for staging buffer anymore
    device->GetLogicalDevice().destroyBuffer(stagingBuffer);
    device->GetLogicalDevice().freeMemory(stagingBufferMemory); 
}

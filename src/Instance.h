#pragma once

#include <bitset>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "QueueFlags.h"
#include "Device.h"

extern const bool ENABLE_VALIDATION;

class Instance {

public:
    Instance() = delete;
    Instance(const char* applicationName, unsigned int additionalExtensionCount = 0, const char** additionalExtensions = nullptr);

    vk::Instance GetVkInstance();
    vk::PhysicalDevice GetPhysicalDevice();
    const QueueFamilyIndices& GetQueueFamilyIndices() const;
    const vk::SurfaceCapabilitiesKHR& GetSurfaceCapabilities() const;
    const std::vector<vk::SurfaceFormatKHR>& GetSurfaceFormats() const;
    const std::vector<vk::PresentModeKHR>& GetPresentModes() const;
    
    uint32_t GetMemoryTypeIndex(uint32_t types, vk::MemoryPropertyFlags properties) const;
    vk::Format GetSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const;

    void PickPhysicalDevice(std::vector<const char*> deviceExtensions, QueueFlagBits requiredQueues, const vk::SurfaceKHR& surface = nullptr);

    Device* CreateDevice(QueueFlagBits requiredQueues, vk::PhysicalDeviceFeatures deviceFeatures);

    ~Instance();

private:

    void initDebugReport();

    vk::UniqueInstance instance;
    VkDebugReportCallbackEXT debugReportCallback;
    std::vector<const char*> deviceExtensions;
    vk::PhysicalDevice physicalDevice;
    QueueFamilyIndices queueFamilyIndices;
    vk::SurfaceCapabilitiesKHR surfaceCapabilities;
    std::vector<vk::SurfaceFormatKHR> surfaceFormats;
    std::vector<vk::PresentModeKHR> presentModes;
    vk::PhysicalDeviceMemoryProperties deviceMemoryProperties;
};

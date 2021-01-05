#pragma once

#include <array>
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include "QueueFlags.h"
#include "SwapChain.h"

class SwapChain;
class Device {
    friend class Instance;

public:
    SwapChain* CreateSwapChain(vk::SurfaceKHR surface, unsigned int numBuffers);
    Instance* GetInstance();
    vk::Device GetLogicalDevice();
    vk::Queue GetQueue(QueueFlags flag);
    unsigned int GetQueueIndex(QueueFlags flag);
    ~Device();

private:
    using Queues = std::array<vk::Queue, sizeof(QueueFlags)>;
    
    Device() = delete;
    Device(Instance* instance, vk::Device device, Queues queues);

    Instance* instance;
    vk::Device logicalDevice;
    Queues queues;
};

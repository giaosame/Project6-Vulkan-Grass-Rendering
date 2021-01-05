#pragma once
#include <vector>
#include "Device.h"

class Device;
class SwapChain {
    friend class Device;

public:
    vk::SwapchainKHR GetVkSwapChain() const;
    vk::Format GetVkImageFormat() const;
    vk::Extent2D GetVkExtent() const;
    uint32_t GetIndex() const;
    uint32_t GetCount() const;
    vk::Image GetVkImage(uint32_t index) const;
    vk::Semaphore GetImageAvailableVkSemaphore() const;
    vk::Semaphore GetRenderFinishedVkSemaphore() const;
    
    void Recreate();
    bool Acquire();
    bool Present();
    ~SwapChain();

private:
    SwapChain(Device* device, vk::SurfaceKHR vkSurface, unsigned int numBuffers);
    void Create();
    void Destroy();

    Device* device;
    vk::SurfaceKHR vkSurface;
    unsigned int numBuffers;
    vk::SwapchainKHR vkSwapChain;
    std::vector<vk::Image> vkSwapChainImages;
    vk::Format vkSwapChainImageFormat;
    vk::Extent2D vkSwapChainExtent;
    uint32_t imageIndex = 0;

    vk::Semaphore imageAvailableSemaphore;
    vk::Semaphore renderFinishedSemaphore;
};

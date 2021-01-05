#include <vector>
#include "SwapChain.h"
#include "Instance.h"
#include "Device.h"
#include "Window.h"

namespace {
  // Specify the color channel format and color space type
  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
      // vk::Format::eUndefined indicates that the surface has no preferred format, so we can choose any
      if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
          return vk::SurfaceFormatKHR{ vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
      }

      // Otherwise, choose a preferred combination
      for (const auto& availableFormat : availableFormats) {
          // Ideal format and color space
          if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
              return availableFormat;
          }
      }

      // Otherwise, return any format
      return availableFormats[0];
  }

  // Specify the presentation mode of the swap chain
  vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes) {
      // Second choice
      vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
      
      for (const auto& availablePresentMode : availablePresentModes) {
          if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
              // First choice
              return availablePresentMode;
          }
          else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
              // Third choice
              bestMode = availablePresentMode;
          }
      }

      return bestMode;
  }

  // Specify the swap extent (resolution) of the swap chain
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) {
      if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
          return capabilities.currentExtent;
      } else {
          int width, height;
          glfwGetWindowSize(window, &width, &height);
          vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

          actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
          actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

          return actualExtent;
      }
  }
}

SwapChain::SwapChain(Device* device, vk::SurfaceKHR vkSurface, unsigned int numBuffers)
  : device(device), vkSurface(vkSurface), numBuffers(numBuffers) 
{    
    Create();

    vk::SemaphoreCreateInfo semaphoreInfo;
    try {
        imageAvailableSemaphore = device->GetLogicalDevice().createSemaphore(semaphoreInfo);
        renderFinishedSemaphore = device->GetLogicalDevice().createSemaphore(semaphoreInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create semaphores");
    }
}

void SwapChain::Create() {
    auto* instance = device->GetInstance();
    const auto& surfaceCapabilities = instance->GetSurfaceCapabilities();

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(instance->GetSurfaceFormats());
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(instance->GetPresentModes());
    vk::Extent2D extent = chooseSwapExtent(surfaceCapabilities, GetGLFWWindow());

    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    imageCount = numBuffers > imageCount ? numBuffers : imageCount;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
        imageCount = surfaceCapabilities.maxImageCount;
    }

    // --- Create swap chain ---
    vk::SwapchainCreateInfoKHR createInfo;
    // Specify surface to be tied to
    createInfo.setSurface(vkSurface);

    // Add details of the swap chain
    createInfo.setMinImageCount(imageCount);
    createInfo.setImageFormat(surfaceFormat.format);
    createInfo.setImageColorSpace(surfaceFormat.colorSpace);
    createInfo.setImageExtent(extent);
    createInfo.setImageArrayLayers(1);
    createInfo.setImageUsage(vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment));

    const auto& queueFamilyIndices = instance->GetQueueFamilyIndices();
    if (queueFamilyIndices[QueueFlags::Graphics] != queueFamilyIndices[QueueFlags::Present]) {
        // Images can be used across multiple queue families without explicit ownership transfers
        createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        createInfo.setQueueFamilyIndexCount(2);
        unsigned int indices[] = {
            static_cast<unsigned int>(queueFamilyIndices[QueueFlags::Graphics]),
            static_cast<unsigned int>(queueFamilyIndices[QueueFlags::Present])
        };
        createInfo.setPQueueFamilyIndices(indices);
    }
    else {
        // An image is owned by one queue family at a time and ownership must be explicitly transfered between uses
        createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        createInfo.setQueueFamilyIndexCount(0);
        createInfo.setPQueueFamilyIndices(nullptr);
    }

    // Specify transform on images in the swap chain (no transformation done here)
    createInfo.setPreTransform(surfaceCapabilities.currentTransform);

    // Specify alpha channel usage (set to be ignored here)
    createInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);

    // Specify presentation mode
    createInfo.setPresentMode(presentMode);

    // Specify whether we can clip pixels that are obscured by other windows
    createInfo.setClipped(VK_TRUE);

    // Reference to old swap chain in case current one becomes invalid
    createInfo.setOldSwapchain(nullptr);

    // Create swap chain
    try {
        vkSwapChain = device->GetLogicalDevice().createSwapchainKHR(createInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create swap chain");
    }

    // --- Retrieve swap chain images ---
    vkSwapChainImages = device->GetLogicalDevice().getSwapchainImagesKHR(vkSwapChain);
    vkSwapChainImageFormat = surfaceFormat.format;
    vkSwapChainExtent = extent;
}

void SwapChain::Destroy() {
    device->GetLogicalDevice().destroySwapchainKHR(vkSwapChain);
}

vk::SwapchainKHR SwapChain::GetVkSwapChain() const {
    return vkSwapChain;
}

vk::Format SwapChain::GetVkImageFormat() const {
    return vkSwapChainImageFormat;
}

vk::Extent2D SwapChain::GetVkExtent() const {
    return vkSwapChainExtent;
}

uint32_t SwapChain::GetIndex() const {
    return imageIndex;
}

uint32_t SwapChain::GetCount() const {
    return static_cast<uint32_t>(vkSwapChainImages.size());
}

vk::Image SwapChain::GetVkImage(uint32_t index) const {
    return vkSwapChainImages[index];
}

vk::Semaphore SwapChain::GetImageAvailableVkSemaphore() const {
    return imageAvailableSemaphore;

}

vk::Semaphore SwapChain::GetRenderFinishedVkSemaphore() const {
    return renderFinishedSemaphore;
}

void SwapChain::Recreate() {
    Destroy();
    Create();
}

bool SwapChain::Acquire() {
    if (ENABLE_VALIDATION) {
        // The validation layer implementation expects the application to explicitly synchronize with the GPU
        device->GetQueue(QueueFlags::Present).waitIdle();
    }
    
    try {
        auto result = device->GetLogicalDevice().acquireNextImageKHR(vkSwapChain, std::numeric_limits<uint64_t>::max(),
                                                                     imageAvailableSemaphore, nullptr);
        imageIndex = result.value;
    }
    catch (vk::OutOfDateKHRError err) {
        Recreate();
        return false;
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    return true;
}

bool SwapChain::Present() {
    std::array<vk::Semaphore, 1> signalSemaphores = { renderFinishedSemaphore };

    // Submit result back to swap chain for presentation
    vk::PresentInfoKHR presentInfo;
    presentInfo.setWaitSemaphoreCount(1);
    presentInfo.setPWaitSemaphores(signalSemaphores.data());

    std::array<vk::SwapchainKHR, 1> swapChains = { vkSwapChain };
    presentInfo.setSwapchainCount(1);
    presentInfo.setPSwapchains(swapChains.data());
    presentInfo.setPImageIndices(&imageIndex);
    presentInfo.setPResults(nullptr);

    vk::Result presentResult;
    try {
        presentResult = device->GetQueue(QueueFlags::Present).presentKHR(presentInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to present swap chain image");
    }

    if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR) {
        Recreate();
        return false;
    }

    return true;
}

SwapChain::~SwapChain() {
    device->GetLogicalDevice().destroySemaphore(imageAvailableSemaphore);
    device->GetLogicalDevice().destroySemaphore(renderFinishedSemaphore);
    Destroy();
}

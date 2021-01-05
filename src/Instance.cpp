
#include <set>
#include <vector>
#include <iostream>
#include "Instance.h"

#ifdef NDEBUG
const bool ENABLE_VALIDATION = false;
#else
const bool ENABLE_VALIDATION = true;
#endif

namespace {
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    // Get the required list of extensions based on whether validation layers are enabled
    std::vector<const char*> getRequiredExtensions() {
        std::vector<const char*> extensions;

        if (ENABLE_VALIDATION) {
            extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        return extensions;
    }

    // Callback function to allow messages from validation layers to be received
    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugReportFlagsEXT flags,
        VkDebugReportObjectTypeEXT objType,
        uint64_t obj,
        size_t location,
        int32_t code,
        const char* layerPrefix,
        const char* msg,
        void *userData) {

        fprintf(stderr, "Validation layer: %s\n", msg);
        return VK_FALSE;
    }
}

Instance::Instance(const char* applicationName, unsigned int additionalExtensionCount, const char** additionalExtensions) {
    // --- Specify details about our application ---
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName(applicationName);
    appInfo.setApplicationVersion(VK_MAKE_VERSION(1, 0, 0));
    appInfo.setPEngineName("No Engine");
    appInfo.setEngineVersion(VK_MAKE_VERSION(1, 0, 0));
    appInfo.setApiVersion(VK_API_VERSION_1_0);
    
    // --- Create Vulkan instance ---
    vk::InstanceCreateInfo createInfo;
    createInfo.setFlags(vk::InstanceCreateFlags());
    createInfo.setPApplicationInfo(&appInfo);

    // Get extensions necessary for Vulkan to interface with GLFW
    auto extensions = getRequiredExtensions();
    for (unsigned int i = 0; i < additionalExtensionCount; ++i) {
        extensions.push_back(additionalExtensions[i]);
    }
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Specify global validation layers
    if (ENABLE_VALIDATION) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    // Create instance
    try {
        instance = vk::createInstanceUnique(createInfo, nullptr);
    } 
    catch (vk::SystemError err) {
        throw std::runtime_error("failed to create instance!");
    }

    try {
        initDebugReport();
    }
    catch (vk::SystemError err) {
        std::cerr << err.what() << std::endl;
    }
}

vk::Instance Instance::GetVkInstance() {
    return instance.get();
}

vk::PhysicalDevice Instance::GetPhysicalDevice() {
    return physicalDevice;
}

const vk::SurfaceCapabilitiesKHR& Instance::GetSurfaceCapabilities() const {
    return surfaceCapabilities;
}

const QueueFamilyIndices& Instance::GetQueueFamilyIndices() const {
    return queueFamilyIndices;
}

const std::vector<vk::SurfaceFormatKHR>& Instance::GetSurfaceFormats() const {
    return surfaceFormats;
}

const std::vector<vk::PresentModeKHR>& Instance::GetPresentModes() const {
    return presentModes;
}

uint32_t Instance::GetMemoryTypeIndex(uint32_t typeBits, vk::MemoryPropertyFlags properties) const {
    // Iterate over all memory types available for the device used in this example
    for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        typeBits >>= 1;
    }
    throw std::runtime_error("Could not find a suitable memory type!");
}

vk::Format Instance::GetSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
    for (vk::Format format : candidates) {
        vk::FormatProperties properties;
        properties = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format");
}

void Instance::initDebugReport() {
    if (ENABLE_VALIDATION) {
        // Specify details for callback
        VkDebugReportCallbackCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
        createInfo.pfnCallback = debugCallback;

        if ([&]() {
            auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(*instance, "vkCreateDebugReportCallbackEXT");
            if (func != nullptr) {
                return func(*instance, &createInfo, nullptr, &debugReportCallback);
            }
            else {
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
        }() != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug callback");
        }
    }
}


namespace {
    QueueFamilyIndices checkDeviceQueueSupport(vk::PhysicalDevice device, QueueFlagBits requiredQueues, const vk::SurfaceKHR surface = nullptr) {
        QueueFamilyIndices indices = {};
        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
        if (queueFamilies.empty()) {
            return indices;
        }

        vk::QueueFlags requiredVulkanQueues(0);
        if (requiredQueues[QueueFlags::Graphics]) {
            requiredVulkanQueues |= vk::QueueFlagBits::eGraphics;
        }
        if (requiredQueues[QueueFlags::Compute]) {
            requiredVulkanQueues |= vk::QueueFlagBits::eCompute;
        }
        if (requiredQueues[QueueFlags::Transfer]) {
            requiredVulkanQueues |= vk::QueueFlagBits::eTransfer;
        }

        indices.fill(-1);
        vk::QueueFlags supportedQueues(0);
        bool needsPresent = requiredQueues[QueueFlags::Present];
        bool presentSupported = false;

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueCount > 0) {
                supportedQueues |= queueFamily.queueFlags;
            }

            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices[QueueFlags::Graphics] = i;
            }

            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
                indices[QueueFlags::Compute] = i;
            }

            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) {
                indices[QueueFlags::Transfer] = i;
            }

            if (needsPresent) {
                vk::Bool32 presentSupport = false;
                presentSupport = device.getSurfaceSupportKHR(i, surface);
                if (queueFamily.queueCount > 0 && presentSupport) {
                    presentSupported = true;
                    indices[QueueFlags::Present] = i;
                }
            }

            if ((requiredVulkanQueues & supportedQueues) == requiredVulkanQueues && (!needsPresent || presentSupported)) {
                break;
            }

            i++;
        }

        return indices;
    }

    // Check the physical device for specified extension support
    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device, std::vector<const char*> requiredExtensions) {
        std::set<std::string> requiredExtensionSet(requiredExtensions.begin(), requiredExtensions.end());

        for (const auto& extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensionSet.erase(extension.extensionName);
        }

        return requiredExtensionSet.empty();
    }
}

void Instance::PickPhysicalDevice(std::vector<const char*> deviceExtensions, QueueFlagBits requiredQueues, const vk::SurfaceKHR& surface) {
    // List the graphics cards on the machine
    auto devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    // Evaluate each GPU and check if it is suitable
    for (const auto& device : devices) {
        bool queueSupport = true;
        queueFamilyIndices = checkDeviceQueueSupport(device, requiredQueues, surface);
        for (unsigned int i = 0; i < requiredQueues.size(); ++i) {
            if (requiredQueues[i]) {
                queueSupport &= (queueFamilyIndices[i] >= 0);
            }
        }

        if (requiredQueues[QueueFlags::Present]) {
            // Get basic surface capabilities
            surfaceCapabilities = device.getSurfaceCapabilitiesKHR(surface);
            
            // Query supported surface formats
            surfaceFormats = device.getSurfaceFormatsKHR(surface);

            // Query supported presentation modes
            presentModes = device.getSurfacePresentModesKHR(surface);
        }

        if (queueSupport &&
            checkDeviceExtensionSupport(device, deviceExtensions) &&
            (!requiredQueues[QueueFlags::Present] || (!surfaceFormats.empty() && !presentModes.empty()))
        ) {
            physicalDevice = device;
            break;
        }
    }

    this->deviceExtensions = deviceExtensions;
    
    if (!physicalDevice) {
        throw std::runtime_error("Failed to find a suitable GPU");
    }

    deviceMemoryProperties = physicalDevice.getMemoryProperties();
}

Device* Instance::CreateDevice(QueueFlagBits requiredQueues, vk::PhysicalDeviceFeatures deviceFeatures) {
    std::set<int> uniqueQueueFamilies;
    bool queueSupport = true;
    for (unsigned int i = 0; i < requiredQueues.size(); ++i) {
        if (requiredQueues[i]) {
            queueSupport &= (queueFamilyIndices[i] >= 0);
            uniqueQueueFamilies.insert(queueFamilyIndices[i]);
        }
    }

    if (!queueSupport) {
        throw std::runtime_error("Device does not support requested queues");
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    for (int queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.setQueueFamilyIndex(queueFamily);
        queueCreateInfo.setQueueCount(1);
        queueCreateInfo.setPQueuePriorities(&queuePriority);
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Create logical device
    vk::DeviceCreateInfo createInfo = {};
    createInfo.setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()));
    createInfo.setPQueueCreateInfos(queueCreateInfos.data());
    createInfo.setPEnabledFeatures(&deviceFeatures);

    // Enable device-specific extensions and validation layers
    createInfo.setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions.size()));
    createInfo.setPpEnabledExtensionNames(deviceExtensions.data());

    if (ENABLE_VALIDATION) {
        createInfo.setEnabledLayerCount(static_cast<uint32_t>(validationLayers.size()));
        createInfo.setPpEnabledLayerNames(validationLayers.data());
    } else {
        createInfo.setEnabledLayerCount(0);
    }

    vk::Device vkDevice;
    // Create logical device
    try {
        vkDevice = physicalDevice.createDevice(createInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create logical device");
    }

    Device::Queues queues;
    for (unsigned int i = 0; i < requiredQueues.size(); ++i) {
        if (requiredQueues[i]) {
            queues[i] = vkDevice.getQueue(queueFamilyIndices[i], 0);
        }
    }

    return new Device(this, vkDevice, queues);
}

Instance::~Instance() {
    if (ENABLE_VALIDATION) {
        auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(*instance, "vkDestroyDebugReportCallbackEXT");
        if (func != nullptr) {
            func(*instance, debugReportCallback, nullptr);
        }
    }
    // vkDestroyInstance(instance, nullptr);
}

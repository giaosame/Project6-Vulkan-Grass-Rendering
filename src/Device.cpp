#include "Device.h"
#include "Instance.h"

Device::Device(Instance* instance, vk::Device device, Queues queues)
  : instance(instance), logicalDevice(device), queues(queues) {
}

Instance* Device::GetInstance() {
    return instance;
}

vk::Device Device::GetLogicalDevice() {
    return logicalDevice;
}

vk::Queue Device::GetQueue(QueueFlags flag) {
    return queues[flag];
}

unsigned int Device::GetQueueIndex(QueueFlags flag) {
    return GetInstance()->GetQueueFamilyIndices()[flag];
}

SwapChain* Device::CreateSwapChain(vk::SurfaceKHR surface, unsigned int numBuffers) {
    return new SwapChain(this, surface, numBuffers);
}

Device::~Device() {
    logicalDevice.destroy();
}

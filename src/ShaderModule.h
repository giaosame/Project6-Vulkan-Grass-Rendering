#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <vector>

namespace ShaderModule {
    vk::ShaderModule Create(const std::vector<char>& code, vk::Device logicalDevice);
    vk::ShaderModule Create(const std::string& filename, vk::Device logicalDevice);
}

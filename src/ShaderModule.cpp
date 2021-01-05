#include <fstream>
#include "ShaderModule.h"

namespace {
    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
        return buffer;
    }
}

// Wrap the shaders in shader modules
vk::ShaderModule ShaderModule::Create(const std::vector<char>& code, vk::Device logicalDevice) {
    vk::ShaderModuleCreateInfo createInfo;
    createInfo.setCodeSize(code.size());
    createInfo.setPCode(reinterpret_cast<const uint32_t*>(code.data()));

    vk::ShaderModule shaderModule;
    try {
        shaderModule = logicalDevice.createShaderModule(createInfo);
    } 
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
}

vk::ShaderModule ShaderModule::Create(const std::string& filename, vk::Device logicalDevice) {
    return ShaderModule::Create(readFile(filename), logicalDevice);
}

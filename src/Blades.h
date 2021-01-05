#pragma once
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <array>
#include "Model.h"

constexpr static unsigned int NUM_BLADES = 1 << 13;
constexpr static float MIN_HEIGHT = 1.3f;
constexpr static float MAX_HEIGHT = 2.5f;
constexpr static float MIN_WIDTH = 0.1f;
constexpr static float MAX_WIDTH = 0.14f;
constexpr static float MIN_BEND = 7.0f;
constexpr static float MAX_BEND = 13.0f;

struct Blade {
    // Position and direction
    glm::vec4 v0;
    // Bezier point and height
    glm::vec4 v1;
    // Physical model guide and width
    glm::vec4 v2;
    // Up vector and stiffness coefficient
    glm::vec4 up;

    // Specify vertex input binding description
    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription;
        bindingDescription.setBinding(0);
        bindingDescription.setStride(sizeof(Blade));
        bindingDescription.setInputRate(vk::VertexInputRate::eVertex);

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 4> attributeDescriptions = {};

        // v0
        attributeDescriptions[0].setBinding(0);
        attributeDescriptions[0].setLocation(0);
        attributeDescriptions[0].setFormat(vk::Format::eR32G32B32A32Sfloat);
        attributeDescriptions[0].setOffset(offsetof(Blade, v0));

        // v1
        attributeDescriptions[1].setBinding(0);
        attributeDescriptions[1].setLocation(1);
        attributeDescriptions[1].setFormat(vk::Format::eR32G32B32A32Sfloat);
        attributeDescriptions[1].setOffset(offsetof(Blade, v1));

        // v2
        attributeDescriptions[2].setBinding(0);
        attributeDescriptions[2].setLocation(2);
        attributeDescriptions[2].setFormat(vk::Format::eR32G32B32A32Sfloat);
        attributeDescriptions[2].setOffset(offsetof(Blade, v1));

        // up
        attributeDescriptions[3].setBinding(0);
        attributeDescriptions[3].setLocation(3);
        attributeDescriptions[3].setFormat(vk::Format::eR32G32B32A32Sfloat);
        attributeDescriptions[3].setOffset(offsetof(Blade, up));

        return attributeDescriptions;
    }
};

struct BladeDrawIndirect {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
};

class Blades : public Model {
private:
    vk::Buffer bladesBuffer;
    vk::Buffer culledBladesBuffer;
    vk::Buffer numBladesBuffer;

    vk::DeviceMemory bladesBufferMemory;
    vk::DeviceMemory culledBladesBufferMemory;
    vk::DeviceMemory numBladesBufferMemory;

public:
    Blades(Device* device, vk::CommandPool commandPool, float planeDim);
    vk::Buffer GetBladesBuffer() const;
    vk::Buffer GetCulledBladesBuffer() const;
    vk::Buffer GetNumBladesBuffer() const;
    ~Blades();
};

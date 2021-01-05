#pragma once
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <vector>
#include "Vertex.h"
#include "Device.h"

struct ModelBufferObject {
    glm::mat4 modelMatrix;
};

class Model {
protected:
    Device* device;

    std::vector<Vertex> vertices;
    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;

    std::vector<uint32_t> indices;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    vk::Buffer modelBuffer;
    vk::DeviceMemory modelBufferMemory;
    ModelBufferObject modelBufferObject;

    vk::Image texture;
    vk::ImageView textureView;
    vk::Sampler textureSampler;

public:
    Model() = delete;
    Model(Device* device, vk::CommandPool commandPool, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);
    virtual ~Model();

    void SetTexture(vk::Image texture);

    const std::vector<Vertex>& getVertices() const;

    vk::Buffer getVertexBuffer() const;

    const std::vector<uint32_t>& getIndices() const;

    vk::Buffer getIndexBuffer() const;

    const ModelBufferObject& getModelBufferObject() const;

    vk::Buffer GetModelBuffer() const;
    vk::ImageView GetTextureView() const;
    vk::Sampler GetTextureSampler() const;
};

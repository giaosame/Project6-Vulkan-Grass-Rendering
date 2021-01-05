#include "Model.h"
#include "BufferUtils.h"
#include "Image.h"

Model::Model(Device* device, vk::CommandPool commandPool, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
  : device(device), vertices(vertices), indices(indices) 
{
    if (vertices.size() > 0) {
        BufferUtils::CreateBufferFromData(device, commandPool, this->vertices.data(), vertices.size() * sizeof(Vertex), vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer, vertexBufferMemory);
    }

    if (indices.size() > 0) {
        BufferUtils::CreateBufferFromData(device, commandPool, this->indices.data(), indices.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer, indexBufferMemory);
    }

    modelBufferObject.modelMatrix = glm::mat4(1.0f);
    BufferUtils::CreateBufferFromData(device, commandPool, &modelBufferObject, sizeof(ModelBufferObject), vk::BufferUsageFlagBits::eUniformBuffer, modelBuffer, modelBufferMemory);
}

Model::~Model() {
    if (indices.size() > 0) {
        device->GetLogicalDevice().destroyBuffer(indexBuffer);
        device->GetLogicalDevice().freeMemory(indexBufferMemory);
    }

    if (vertices.size() > 0) {
        device->GetLogicalDevice().destroyBuffer(vertexBuffer);
        device->GetLogicalDevice().freeMemory(vertexBufferMemory);
    }

    device->GetLogicalDevice().destroyBuffer(modelBuffer);
    device->GetLogicalDevice().freeMemory(modelBufferMemory);

    if (textureView) {
        device->GetLogicalDevice().destroyImageView(textureView);
    }

    if (textureSampler) {
        device->GetLogicalDevice().destroySampler(textureSampler);
    }
}

void Model::SetTexture(vk::Image texture) {
    this->texture = texture;
    this->textureView = Image::CreateView(device, texture, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);

    // --- Specify all filters and transformations ---
    vk::SamplerCreateInfo samplerInfo;
   
    // Interpolation of texels that are magnified or minified
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);

    // Addressing mode
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);

    // Anisotropic filtering
    samplerInfo.setAnisotropyEnable(VK_TRUE);
    samplerInfo.setMaxAnisotropy(16);

    // Border color
    samplerInfo.setBorderColor(vk::BorderColor::eIntOpaqueBlack);

    // Choose coordinate system for addressing texels --> [0, 1) here
    samplerInfo.setUnnormalizedCoordinates(VK_FALSE);

    // Comparison function used for filtering operations
    samplerInfo.setCompareEnable(VK_FALSE);
    samplerInfo.setCompareOp(vk::CompareOp::eAlways);

    // Mipmapping
    samplerInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
    samplerInfo.setMipLodBias(0.0f);
    samplerInfo.setMinLod(0.0f);
    samplerInfo.setMaxLod(0.0f);

    try {
        textureSampler = device->GetLogicalDevice().createSampler(samplerInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create texture sampler");
    }
}

const std::vector<Vertex>& Model::getVertices() const {
    return vertices;
}

vk::Buffer Model::getVertexBuffer() const {
    return vertexBuffer;
}

const std::vector<uint32_t>& Model::getIndices() const {
    return indices;
}

vk::Buffer Model::getIndexBuffer() const {
    return indexBuffer;
}

const ModelBufferObject& Model::getModelBufferObject() const {
    return modelBufferObject;
}

vk::Buffer Model::GetModelBuffer() const {
    return modelBuffer;
}

vk::ImageView Model::GetTextureView() const {
    return textureView;
}

vk::Sampler Model::GetTextureSampler() const {
    return textureSampler;
}

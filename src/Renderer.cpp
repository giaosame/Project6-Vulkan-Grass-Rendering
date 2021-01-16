#include "Renderer.h"
#include "Instance.h"
#include "ShaderModule.h"
#include "Vertex.h"
#include "Blades.h"
#include "Camera.h"
#include "Image.h"

static constexpr unsigned int WORKGROUP_SIZE = 32;

Renderer::Renderer(Device* device, SwapChain* swapChain, Scene* scene, Camera* camera)
  : device(device),
    logicalDevice(device->GetLogicalDevice()),
    swapChain(swapChain),
    scene(scene),
    camera(camera) {

    CreateCommandPools();
    CreateRenderPass();
    CreateCameraDescriptorSetLayout();
    CreateModelDescriptorSetLayout();
    CreateTimeDescriptorSetLayout();
    CreateComputeDescriptorSetLayout();
    CreateDescriptorPool();
    CreateCameraDescriptorSet();
    CreateModelDescriptorSets();
    CreateGrassDescriptorSets();
    CreateTimeDescriptorSet();
    CreateComputeDescriptorSets();
    CreateFrameResources();
    CreateGraphicsPipeline();
    CreateGrassPipeline();
    CreateComputePipeline();
    RecordCommandBuffers();
    RecordComputeCommandBuffer();
}

void Renderer::CreateCommandPools() {
    vk::CommandPoolCreateInfo graphicsPoolInfo;
    
    graphicsPoolInfo.setQueueFamilyIndex(device->GetInstance()->GetQueueFamilyIndices()[QueueFlags::Graphics]);
    graphicsPoolInfo.setFlags(vk::CommandPoolCreateFlags(0));

    try {
        graphicsCommandPool = logicalDevice.createCommandPool(graphicsPoolInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create graphics command pool");
    }

    vk::CommandPoolCreateInfo computePoolInfo;
    computePoolInfo.setQueueFamilyIndex(device->GetInstance()->GetQueueFamilyIndices()[QueueFlags::Compute]);
    computePoolInfo.setFlags(vk::CommandPoolCreateFlags(0));

    try {
        computeCommandPool = logicalDevice.createCommandPool(computePoolInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create compute command pool");
    }
}

void Renderer::CreateRenderPass() {
    // Color buffer attachment represented by one of the images from the swap chain
    vk::AttachmentDescription colorAttachment;
    colorAttachment.setFormat(swapChain->GetVkImageFormat());
    colorAttachment.setSamples(vk::SampleCountFlagBits::e1);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
    colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
    colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    // Create a color attachment reference to be used with subpass
    vk::AttachmentReference colorAttachmentRef;
    colorAttachmentRef.setAttachment(0);
    colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    // Depth buffer attachment
    vk::Format depthFormat = device->GetInstance()->GetSupportedFormat({ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint }, 
                                                                        vk::ImageTiling::eOptimal, 
                                                                        vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eDepthStencilAttachment));
    vk::AttachmentDescription depthAttachment;
    depthAttachment.setFormat(depthFormat);
    depthAttachment.setSamples(vk::SampleCountFlagBits::e1);
    depthAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    depthAttachment.setStoreOp(vk::AttachmentStoreOp::eDontCare);
    depthAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    depthAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
    depthAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
    depthAttachment.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // Create a depth attachment reference
    vk::AttachmentReference depthAttachmentRef;
    depthAttachmentRef.setAttachment(1);
    depthAttachmentRef.setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // Create subpass description
    vk::SubpassDescription subpass;
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
    subpass.setColorAttachmentCount(1);
    subpass.setPColorAttachments(&colorAttachmentRef);
    subpass.setPDepthStencilAttachment(&depthAttachmentRef);

    std::array<vk::AttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

    // Specify subpass dependency
    vk::SubpassDependency dependency;
    dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
    dependency.setDstSubpass(0);
    dependency.setSrcStageMask(vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput));
    dependency.setSrcAccessMask(vk::AccessFlags(0));
    dependency.setDstStageMask(vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput));
    dependency.setDstAccessMask(vk::AccessFlags(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite));

    // Create render pass
    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.setAttachmentCount(static_cast<uint32_t>(attachments.size()));
    renderPassInfo.setPAttachments(attachments.data());
    renderPassInfo.setSubpassCount(1);
    renderPassInfo.setPSubpasses(&subpass);
    renderPassInfo.setDependencyCount(1);
    renderPassInfo.setPDependencies(&dependency);

    try {
        renderPass = logicalDevice.createRenderPass(renderPassInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create render pass");
    }
}

void Renderer::CreateCameraDescriptorSetLayout() {
    // Describe the binding of the descriptor set layout
    vk::DescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.setBinding(0);
    uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    uboLayoutBinding.setDescriptorCount(1);
    uboLayoutBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eAll));
    uboLayoutBinding.setPImmutableSamplers(nullptr);

    std::vector<vk::DescriptorSetLayoutBinding> bindings = { uboLayoutBinding };

    // Create the descriptor set layout
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindingCount(static_cast<uint32_t>(bindings.size()));
    layoutInfo.setPBindings(bindings.data());

    try {
        cameraDescriptorSetLayout = logicalDevice.createDescriptorSetLayout(layoutInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create camera descriptor set layout");
    }
}

void Renderer::CreateModelDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.setBinding(0);
    uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    uboLayoutBinding.setDescriptorCount(1);
    uboLayoutBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eVertex));
    uboLayoutBinding.setPImmutableSamplers(nullptr);

    vk::DescriptorSetLayoutBinding samplerLayoutBinding;
    samplerLayoutBinding.setBinding(1);
    samplerLayoutBinding.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
    samplerLayoutBinding.setDescriptorCount(1);
    samplerLayoutBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eFragment));
    samplerLayoutBinding.setPImmutableSamplers(nullptr);

    std::vector<vk::DescriptorSetLayoutBinding> bindings = { uboLayoutBinding, samplerLayoutBinding };

    // Create the descriptor set layout
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindingCount(static_cast<uint32_t>(bindings.size()));
    layoutInfo.setPBindings(bindings.data());

    try {
        modelDescriptorSetLayout = logicalDevice.createDescriptorSetLayout(layoutInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create model descriptor set layout");
    }
}

void Renderer::CreateTimeDescriptorSetLayout() {
    // Describe the binding of the descriptor set layout
    vk::DescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.setBinding(0);
    uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    uboLayoutBinding.setDescriptorCount(1);
    uboLayoutBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eCompute));
    uboLayoutBinding.setPImmutableSamplers(nullptr);

    std::vector<vk::DescriptorSetLayoutBinding> bindings = { uboLayoutBinding };

    // Create the descriptor set layout
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindingCount(static_cast<uint32_t>(bindings.size()));
    layoutInfo.setPBindings(bindings.data());

    try {
        timeDescriptorSetLayout = logicalDevice.createDescriptorSetLayout(layoutInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create time descriptor set layout");
    }
}

void Renderer::CreateComputeDescriptorSetLayout() {
    // Create the descriptor set layout for the compute pipeline
    // This's like a class definition stating why types of information will be stored at each binding
    vk::DescriptorSetLayoutBinding bladesBinding;
    bladesBinding.setBinding(0);
    bladesBinding.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    bladesBinding.setDescriptorCount(1);
    bladesBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eCompute));
    bladesBinding.setPImmutableSamplers(nullptr);

    vk::DescriptorSetLayoutBinding culledBladesBinding;
    culledBladesBinding.setBinding(1);
    culledBladesBinding.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    culledBladesBinding.setDescriptorCount(1);
    culledBladesBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eCompute));
    culledBladesBinding.setPImmutableSamplers(nullptr);

    vk::DescriptorSetLayoutBinding numBladesBinding;
    numBladesBinding.setBinding(2);
    numBladesBinding.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    numBladesBinding.setDescriptorCount(1);
    numBladesBinding.setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eCompute));
    numBladesBinding.setPImmutableSamplers(nullptr);

    std::array<vk::DescriptorSetLayoutBinding, 3> bindings = { bladesBinding, culledBladesBinding, numBladesBinding };
    vk::DescriptorSetLayoutCreateInfo layoutCreateInfo;
    layoutCreateInfo.setBindingCount(static_cast<uint32_t>(bindings.size()));
    layoutCreateInfo.setPBindings(bindings.data());

    try {
        computeDescriptorSetLayout = device->GetLogicalDevice().createDescriptorSetLayout(layoutCreateInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create compute descriptor set layout");
    }
}

void Renderer::CreateDescriptorPool() {
    // Describe which descriptor types that the descriptor sets will contain
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        // Camera
        { vk::DescriptorType::eUniformBuffer, 1},

        // Models + Blades
        { vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(scene->GetModels().size() + scene->GetBlades().size()) },

        // Models + Blades
        { vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(scene->GetModels().size() + scene->GetBlades().size()) },

        // Time (compute)
        { vk::DescriptorType::eUniformBuffer, 1 },

        // TODO: Add any additional types and counts of descriptors you will need to allocate
        // Blades, culledBlades, numBlades aftering compute shader
        { vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(3 * scene->GetBlades().size()) }
    };

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()));
    poolInfo.setPPoolSizes(poolSizes.data());
    poolInfo.setMaxSets(15);

    try {
        descriptorPool = logicalDevice.createDescriptorPool(poolInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void Renderer::CreateCameraDescriptorSet() {
    // Describe the desciptor set
    std::array<vk::DescriptorSetLayout, 1> layouts = { cameraDescriptorSetLayout };
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.setDescriptorPool(descriptorPool);
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPSetLayouts(layouts.data());

    // Allocate descriptor sets
    try {
        logicalDevice.allocateDescriptorSets(&allocInfo, &cameraDescriptorSet);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate camera descriptor set");
    }

    // Configure the descriptors to refer to buffers
    vk::DescriptorBufferInfo cameraBufferInfo;
    cameraBufferInfo.setBuffer(camera->GetBuffer());
    cameraBufferInfo.setOffset(0);
    cameraBufferInfo.setRange(sizeof(CameraBufferObject));

    std::array<vk::WriteDescriptorSet, 1> descriptorWrites;
    descriptorWrites[0].setDstSet(cameraDescriptorSet);
    descriptorWrites[0].setDstBinding(0);
    descriptorWrites[0].setDstArrayElement(0);
    descriptorWrites[0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
    descriptorWrites[0].setDescriptorCount(1);
    descriptorWrites[0].setPBufferInfo(&cameraBufferInfo);
    descriptorWrites[0].setPImageInfo(nullptr);
    descriptorWrites[0].setPTexelBufferView(nullptr);
   
    // Update descriptor sets
    logicalDevice.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateModelDescriptorSets() {
    modelDescriptorSets.resize(scene->GetModels().size());

    // Describe the desciptor set
    std::array<vk::DescriptorSetLayout, 1> layouts = { modelDescriptorSetLayout };

    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.setDescriptorPool(descriptorPool);
    allocInfo.setDescriptorSetCount(static_cast<uint32_t>(modelDescriptorSets.size()));
    allocInfo.setPSetLayouts(layouts.data());
   
    // Allocate descriptor sets
    try {
        modelDescriptorSets = logicalDevice.allocateDescriptorSets(allocInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    std::vector<vk::WriteDescriptorSet> descriptorWrites(2 * modelDescriptorSets.size());

    for (uint32_t i = 0; i < scene->GetModels().size(); ++i) {
        vk::DescriptorBufferInfo modelBufferInfo;
        modelBufferInfo.setBuffer(scene->GetModels()[i]->GetModelBuffer());
        modelBufferInfo.setOffset(0);
        modelBufferInfo.setRange(sizeof(ModelBufferObject));

        // Bind image and sampler resources to the descriptor
        vk::DescriptorImageInfo imageInfo;
        imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        imageInfo.setImageView(scene->GetModels()[i]->GetTextureView());
        imageInfo.setSampler(scene->GetModels()[i]->GetTextureSampler());
        
        descriptorWrites[2 * i + 0].setDstSet(modelDescriptorSets[i]);
        descriptorWrites[2 * i + 0].setDstBinding(0);
        descriptorWrites[2 * i + 0].setDstArrayElement(0);
        descriptorWrites[2 * i + 0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
        descriptorWrites[2 * i + 0].setDescriptorCount(1);
        descriptorWrites[2 * i + 0].setPBufferInfo(&modelBufferInfo);
        descriptorWrites[2 * i + 0].setPImageInfo(nullptr);
        descriptorWrites[2 * i + 0].setPTexelBufferView(nullptr);
        
        descriptorWrites[2 * i + 1].setDstSet(modelDescriptorSets[i]);
        descriptorWrites[2 * i + 1].setDstBinding(1);
        descriptorWrites[2 * i + 1].setDstArrayElement(0);
        descriptorWrites[2 * i + 1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
        descriptorWrites[2 * i + 1].setDescriptorCount(1);
        descriptorWrites[2 * i + 1].setPImageInfo(&imageInfo);
    }

    // Update descriptor sets
    logicalDevice.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateGrassDescriptorSets() {
    grassDescriptorSets.resize(scene->GetBlades().size());
    // Create Descriptor sets for the grass.
    // This should involve creating descriptor sets which point to the model matrix of each group of grass blades
    std::array<vk::DescriptorSetLayout, 1> layouts = { modelDescriptorSetLayout };

    vk::DescriptorSetAllocateInfo allocateInfo;
    allocateInfo.setDescriptorPool(descriptorPool);
    allocateInfo.setDescriptorSetCount(static_cast<uint32_t>(grassDescriptorSets.size()));
    allocateInfo.setPSetLayouts(layouts.data());

    // Allocate descriptor sets
    try {
        grassDescriptorSets = logicalDevice.allocateDescriptorSets(allocateInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate grass descriptor sets");
    }

    std::vector<vk::WriteDescriptorSet> descriptorWrites(grassDescriptorSets.size());
    for (int i = 0; i < scene->GetBlades().size(); i++)
    {
        vk::DescriptorBufferInfo modelBufferInfo;
        modelBufferInfo.setBuffer(scene->GetBlades()[i]->GetModelBuffer());
        modelBufferInfo.setOffset(0);
        modelBufferInfo.setRange(sizeof(ModelBufferObject));

        descriptorWrites[i].setDstSet(grassDescriptorSets[i]);
        descriptorWrites[i].setDstBinding(0);
        descriptorWrites[i].setDstArrayElement(0);
        descriptorWrites[i].setDescriptorType(vk::DescriptorType::eUniformBuffer);
        descriptorWrites[i].setDescriptorCount(1);
        descriptorWrites[i].setPBufferInfo(&modelBufferInfo);
        descriptorWrites[i].setPImageInfo(nullptr);
        descriptorWrites[i].setPTexelBufferView(nullptr);
    }

    logicalDevice.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateTimeDescriptorSet() {
    // Describe the desciptor set
    std::array<vk::DescriptorSetLayout, 1> layouts = { timeDescriptorSetLayout };
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.setDescriptorPool(descriptorPool);
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPSetLayouts(layouts.data());
  
    // Allocate descriptor sets
    try {
        logicalDevice.allocateDescriptorSets(&allocInfo, &timeDescriptorSet);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to time allocate descriptor set");
    }

    // Configure the descriptors to refer to buffers
    vk::DescriptorBufferInfo timeBufferInfo;
    timeBufferInfo.setBuffer(scene->GetTimeBuffer());
    timeBufferInfo.setOffset(0);
    timeBufferInfo.setRange(sizeof(Time));

    std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {};
    descriptorWrites[0].setDstSet(timeDescriptorSet);
    descriptorWrites[0].setDstBinding(0);
    descriptorWrites[0].setDstArrayElement(0);
    descriptorWrites[0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
    descriptorWrites[0].setDescriptorCount(1);
    descriptorWrites[0].setPBufferInfo(&timeBufferInfo);
    descriptorWrites[0].setPImageInfo(nullptr);
    descriptorWrites[0].setPTexelBufferView(nullptr);
  
    // Update descriptor sets
    logicalDevice.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateComputeDescriptorSets() {
    computeDescriptorSets.resize(scene->GetBlades().size());
    // Create Descriptor sets for the compute pipeline
    // The descriptors should point to Storage buffers which will hold the grass blades, the culled grass blades, and the output number of grass blades 
    std::array<vk::DescriptorSetLayout, 1> layouts = { computeDescriptorSetLayout };
    vk::DescriptorSetAllocateInfo allocateInfo;
    allocateInfo.setDescriptorPool(descriptorPool);
    allocateInfo.setDescriptorSetCount(static_cast<uint32_t>(computeDescriptorSets.size()));
    allocateInfo.setPSetLayouts(layouts.data());

    try {
        computeDescriptorSets = logicalDevice.allocateDescriptorSets(allocateInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate compute descriptor set");
    }

    std::vector<vk::WriteDescriptorSet> computeDescriptorWrites;
    for (int i = 0; i < scene->GetBlades().size(); i++) {
        // Bind and write blades buffer to its descriptor
        vk::DescriptorBufferInfo bladesBufferInfo;
        bladesBufferInfo.setBuffer(scene->GetBlades()[i]->GetBladesBuffer());
        bladesBufferInfo.setOffset(0);
        bladesBufferInfo.setRange(static_cast<uint32_t>(NUM_BLADES * sizeof(Blade)));

        vk::WriteDescriptorSet bladesDescriptorWrite;
        bladesDescriptorWrite.setDstSet(computeDescriptorSets[i]);
        bladesDescriptorWrite.setDstBinding(0);
        bladesDescriptorWrite.setDstArrayElement(0);
        bladesDescriptorWrite.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        bladesDescriptorWrite.setDescriptorCount(1);
        bladesDescriptorWrite.setPBufferInfo(&bladesBufferInfo);
        bladesDescriptorWrite.setPImageInfo(nullptr);
        bladesDescriptorWrite.setPTexelBufferView(nullptr);
        
        // Bind and write culled blades buffer to its descriptor
        vk::DescriptorBufferInfo culledBladesBufferInfo;
        culledBladesBufferInfo.setBuffer(scene->GetBlades()[i]->GetCulledBladesBuffer());
        culledBladesBufferInfo.setOffset(0);
        culledBladesBufferInfo.setRange(static_cast<uint32_t>(NUM_BLADES * sizeof(Blade)));

        vk::WriteDescriptorSet culledBladesDescriptorWrite;
        culledBladesDescriptorWrite.setDstSet(computeDescriptorSets[i]);
        culledBladesDescriptorWrite.setDstBinding(1);
        culledBladesDescriptorWrite.setDstArrayElement(0);
        culledBladesDescriptorWrite.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        culledBladesDescriptorWrite.setDescriptorCount(1);
        culledBladesDescriptorWrite.setPBufferInfo(&culledBladesBufferInfo);
        culledBladesDescriptorWrite.setPImageInfo(nullptr);
        culledBladesDescriptorWrite.setPTexelBufferView(nullptr);

        // Bind and write num blades buffer to its descriptor
        vk::DescriptorBufferInfo numBladesBufferInfo;
        numBladesBufferInfo.setBuffer(scene->GetBlades()[i]->GetNumBladesBuffer());
        numBladesBufferInfo.setOffset(0);
        numBladesBufferInfo.setRange(static_cast<uint32_t>(sizeof(BladeDrawIndirect)));

        vk::WriteDescriptorSet numBladesDescriptorWrite;
        numBladesDescriptorWrite.setDstSet(computeDescriptorSets[i]);
        numBladesDescriptorWrite.setDstBinding(2);
        numBladesDescriptorWrite.setDstArrayElement(0);
        numBladesDescriptorWrite.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        numBladesDescriptorWrite.setDescriptorCount(1);
        numBladesDescriptorWrite.setPBufferInfo(&numBladesBufferInfo);
        numBladesDescriptorWrite.setPImageInfo(nullptr);
        numBladesDescriptorWrite.setPTexelBufferView(nullptr);

        computeDescriptorWrites.push_back(bladesDescriptorWrite);
        computeDescriptorWrites.push_back(culledBladesDescriptorWrite);
        computeDescriptorWrites.push_back(numBladesDescriptorWrite);
    }

    logicalDevice.updateDescriptorSets(static_cast<uint32_t>(computeDescriptorWrites.size()), computeDescriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateGraphicsPipeline() {
    vk::ShaderModule vertShaderModule = ShaderModule::Create("shaders/graphics.vert.spv", logicalDevice);
    vk::ShaderModule fragShaderModule = ShaderModule::Create("shaders/graphics.frag.spv", logicalDevice);

    // Assign each shader module to the appropriate stage in the pipeline
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
    vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex);
    vertShaderStageInfo.setModule(vertShaderModule);
    vertShaderStageInfo.setPName("main");

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
    fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
    fragShaderStageInfo.setModule(fragShaderModule);
    fragShaderStageInfo.setPName("main");

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

    // --- Set up fixed-function stages ---

    // Vertex input
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.setVertexBindingDescriptionCount(1);
    vertexInputInfo.setPVertexBindingDescriptions(&bindingDescription);
    vertexInputInfo.setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()));
    vertexInputInfo.setPVertexAttributeDescriptions(attributeDescriptions.data());

    // Input assembly
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
    inputAssembly.setPrimitiveRestartEnable(VK_FALSE);

    // Viewports and Scissors (rectangles that define in which regions pixels are stored)
    vk::Viewport viewport;
    viewport.setX(0.0f);
    viewport.setY(0.0f);
    viewport.setWidth(static_cast<float>(swapChain->GetVkExtent().width));
    viewport.setHeight(static_cast<float>(swapChain->GetVkExtent().height));
    viewport.setMinDepth(0.0f);
    viewport.setMaxDepth(1.0f);

    vk::Rect2D scissor;
    scissor.setOffset({ 0, 0 });
    scissor.setExtent(swapChain->GetVkExtent());

    vk::PipelineViewportStateCreateInfo viewportState;
    viewportState.setViewportCount(1);
    viewportState.setPViewports(&viewport);
    viewportState.setScissorCount(1);
    viewportState.setPScissors(&scissor);

    // Rasterizer
    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.setDepthClampEnable(VK_FALSE);
    rasterizer.setRasterizerDiscardEnable(VK_FALSE);
    rasterizer.setPolygonMode(vk::PolygonMode::eFill);
    rasterizer.setLineWidth(1.0f);
    rasterizer.setCullMode(vk::CullModeFlags(vk::CullModeFlagBits::eNone));
    rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);
    rasterizer.setDepthBiasEnable(VK_FALSE);
    rasterizer.setDepthBiasConstantFactor(0.0f);
    rasterizer.setDepthBiasClamp(0.0f);
    rasterizer.setDepthBiasSlopeFactor(0.0f);

    // Multisampling (turned off here)
    vk::PipelineMultisampleStateCreateInfo multisampling;
    multisampling.setSampleShadingEnable(VK_FALSE);
    multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    multisampling.setMinSampleShading(1.0f);
    multisampling.setPSampleMask(nullptr);
    multisampling.setAlphaToCoverageEnable(VK_FALSE);
    multisampling.setAlphaToOneEnable(VK_FALSE);

    // Depth testing
    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.setDepthTestEnable(VK_TRUE);
    depthStencil.setDepthWriteEnable(VK_TRUE);
    depthStencil.setDepthCompareOp(vk::CompareOp::eLess);
    depthStencil.setDepthBoundsTestEnable(VK_FALSE);
    depthStencil.setMinDepthBounds(0.0f);
    depthStencil.setMaxDepthBounds(1.0f);
    depthStencil.setStencilTestEnable(VK_FALSE);

    // Color blending (turned off here, but showing options for learning)
    // --> Configuration per attached framebuffer
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlags(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA));
    colorBlendAttachment.setBlendEnable(VK_FALSE);
    colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eOne);
    colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eZero);
    colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd);
    colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
    colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
    colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd);

    // --> Global color blending settings
    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.setLogicOpEnable(VK_FALSE);
    colorBlending.setLogicOp(vk::LogicOp::eCopy);
    colorBlending.setAttachmentCount(1);
    colorBlending.setPAttachments(&colorBlendAttachment);
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts = { cameraDescriptorSetLayout, modelDescriptorSetLayout };

    // Pipeline layout: used to specify uniform values
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayoutCount(static_cast<uint32_t>(descriptorSetLayouts.size()));
    pipelineLayoutInfo.setPSetLayouts(descriptorSetLayouts.data());
    pipelineLayoutInfo.setPushConstantRangeCount(0);
    pipelineLayoutInfo.setPushConstantRanges(0);

    try {
        graphicsPipelineLayout = logicalDevice.createPipelineLayout(pipelineLayoutInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // --- Create graphics pipeline ---
    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.setStageCount(2);
    pipelineInfo.setPStages(shaderStages.data());
    pipelineInfo.setPVertexInputState(&vertexInputInfo);
    pipelineInfo.setPInputAssemblyState(&inputAssembly);
    pipelineInfo.setPViewportState(&viewportState);
    pipelineInfo.setPRasterizationState(&rasterizer);
    pipelineInfo.setPMultisampleState(&multisampling);
    pipelineInfo.setPDepthStencilState(&depthStencil);
    pipelineInfo.setPColorBlendState(&colorBlending);
    pipelineInfo.setPDynamicState(nullptr);
    pipelineInfo.setLayout(graphicsPipelineLayout);
    pipelineInfo.setRenderPass(renderPass);
    pipelineInfo.setSubpass(0);
    pipelineInfo.setBasePipelineHandle(nullptr);
    pipelineInfo.setBasePipelineIndex(1);

    try {
        graphicsPipeline = (vk::Pipeline)logicalDevice.createGraphicsPipeline(nullptr, pipelineInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    logicalDevice.destroyShaderModule(vertShaderModule);
    logicalDevice.destroyShaderModule(fragShaderModule);
}

void Renderer::CreateGrassPipeline() {
    // --- Set up programmable shaders ---
    vk::ShaderModule vertShaderModule = ShaderModule::Create("shaders/grass.vert.spv", logicalDevice);
    vk::ShaderModule tescShaderModule = ShaderModule::Create("shaders/grass.tesc.spv", logicalDevice);
    vk::ShaderModule teseShaderModule = ShaderModule::Create("shaders/grass.tese.spv", logicalDevice);
    vk::ShaderModule fragShaderModule = ShaderModule::Create("shaders/grass.frag.spv", logicalDevice);

    // Assign each shader module to the appropriate stage in the pipeline
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
    vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex);
    vertShaderStageInfo.setModule(vertShaderModule);
    vertShaderStageInfo.setPName("main");
  
    vk::PipelineShaderStageCreateInfo tescShaderStageInfo;
    tescShaderStageInfo.setStage(vk::ShaderStageFlagBits::eTessellationControl);
    tescShaderStageInfo.setModule(tescShaderModule);
    tescShaderStageInfo.setPName("main");
  
    vk::PipelineShaderStageCreateInfo teseShaderStageInfo;
    teseShaderStageInfo.setStage(vk::ShaderStageFlagBits::eTessellationEvaluation);
    teseShaderStageInfo.setModule(teseShaderModule);
    teseShaderStageInfo.setPName("main");
   
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
    fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
    fragShaderStageInfo.setModule(fragShaderModule);
    fragShaderStageInfo.setPName("main");
    
    std::array<vk::PipelineShaderStageCreateInfo, 4> shaderStages = { vertShaderStageInfo, tescShaderStageInfo, teseShaderStageInfo, fragShaderStageInfo };

    // --- Set up fixed-function stages ---
    // Vertex input
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
   
    auto bindingDescription = Blade::getBindingDescription();
    auto attributeDescriptions = Blade::getAttributeDescriptions();

    vertexInputInfo.setVertexBindingDescriptionCount(1);
    vertexInputInfo.setPVertexBindingDescriptions(&bindingDescription);
    vertexInputInfo.setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()));
    vertexInputInfo.setPVertexAttributeDescriptions(attributeDescriptions.data());
   
    // Input Assembly
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.setTopology(vk::PrimitiveTopology::ePatchList);
    inputAssembly.setPrimitiveRestartEnable(VK_FALSE);
   
    // Viewports and Scissors (rectangles that define in which regions pixels are stored)
    vk::Viewport viewport;
    viewport.setX(0.0f);
    viewport.setY(0.0f);
    viewport.setWidth(static_cast<float>(swapChain->GetVkExtent().width));
    viewport.setHeight(static_cast<float>(swapChain->GetVkExtent().height));
    viewport.setMinDepth(0.0f);
    viewport.setMaxDepth(1.0f);

    vk::Rect2D scissor = {};
    scissor.setOffset({ 0, 0 });
    scissor.setExtent(swapChain->GetVkExtent());

    vk::PipelineViewportStateCreateInfo viewportState;
    viewportState.setViewportCount(1);
    viewportState.setPViewports(&viewport);
    viewportState.setScissorCount(1);
    viewportState.setPScissors(&scissor);
   
    // Rasterizer
    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.setDepthClampEnable(VK_FALSE);
    rasterizer.setRasterizerDiscardEnable(VK_FALSE);
    rasterizer.setPolygonMode(vk::PolygonMode::eFill);
    rasterizer.setLineWidth(1.0f);
    rasterizer.setCullMode(vk::CullModeFlags(vk::CullModeFlagBits::eNone));
    rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);
    rasterizer.setDepthBiasEnable(VK_FALSE);
    rasterizer.setDepthBiasConstantFactor(0.0f);
    rasterizer.setDepthBiasClamp(0.0f);
    rasterizer.setDepthBiasSlopeFactor(0.0f);
    
    // Multisampling (turned off here)
    vk::PipelineMultisampleStateCreateInfo multisampling;
    multisampling.setSampleShadingEnable(VK_FALSE);
    multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    multisampling.setMinSampleShading(1.0f);
    multisampling.setPSampleMask(nullptr);
    multisampling.setAlphaToCoverageEnable(VK_FALSE);
    multisampling.setAlphaToOneEnable(VK_FALSE);
    
    // Depth testing
    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.setDepthTestEnable(VK_TRUE);
    depthStencil.setDepthWriteEnable(VK_TRUE);
    depthStencil.setDepthCompareOp(vk::CompareOp::eLess);
    depthStencil.setDepthBoundsTestEnable(VK_FALSE);
    depthStencil.setMinDepthBounds(0.0f);
    depthStencil.setMaxDepthBounds(1.0f);
    depthStencil.setStencilTestEnable(VK_FALSE);
    
    // Color blending (turned off here, but showing options for learning)
    // --> Configuration per attached framebuffer
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlags(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                                   vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA));
    colorBlendAttachment.setBlendEnable(VK_FALSE);
    colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eOne);
    colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eZero);
    colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd);
    colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
    colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
    colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd);
    
    // --> Global color blending settings
    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.setLogicOpEnable(VK_FALSE);
    colorBlending.setLogicOp(vk::LogicOp::eCopy);
    colorBlending.setAttachmentCount(1);
    colorBlending.setPAttachments(&colorBlendAttachment);
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts = { cameraDescriptorSetLayout, modelDescriptorSetLayout };

    // Pipeline layout: used to specify uniform values
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayoutCount(static_cast<uint32_t>(descriptorSetLayouts.size()));
    pipelineLayoutInfo.setPSetLayouts(descriptorSetLayouts.data());
    pipelineLayoutInfo.setPushConstantRangeCount(0);
    pipelineLayoutInfo.setPPushConstantRanges(0);
    
    try {
        grassPipelineLayout = logicalDevice.createPipelineLayout(pipelineLayoutInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create grass pipeline layout");
    }

    // Tessellation state
    vk::PipelineTessellationStateCreateInfo tessellationInfo;
    tessellationInfo.setPNext(nullptr);
    tessellationInfo.setFlags(vk::PipelineTessellationStateCreateFlags(0));
    tessellationInfo.setPatchControlPoints(1);
    
    // --- Create graphics pipeline ---
    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.setStageCount(static_cast<uint32_t>(shaderStages.size()));
    pipelineInfo.setPStages(shaderStages.data());
    pipelineInfo.setPVertexInputState(&vertexInputInfo);
    pipelineInfo.setPInputAssemblyState(&inputAssembly);
    pipelineInfo.setPViewportState(&viewportState);
    pipelineInfo.setPRasterizationState(&rasterizer);
    pipelineInfo.setPMultisampleState(&multisampling);
    pipelineInfo.setPDepthStencilState(&depthStencil);
    pipelineInfo.setPColorBlendState(&colorBlending);
    pipelineInfo.setPTessellationState(&tessellationInfo);
    pipelineInfo.setPDynamicState(nullptr);
    pipelineInfo.setLayout(grassPipelineLayout);
    pipelineInfo.setRenderPass(renderPass);
    pipelineInfo.setSubpass(0);
    pipelineInfo.setBasePipelineHandle(nullptr);
    pipelineInfo.setBasePipelineIndex(-1);

    try {
        grassPipeline = (vk::Pipeline)logicalDevice.createGraphicsPipeline(nullptr, pipelineInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    // No need for the shader modules anymore
    logicalDevice.destroyShaderModule(vertShaderModule);
    logicalDevice.destroyShaderModule(tescShaderModule);
    logicalDevice.destroyShaderModule(teseShaderModule);
    logicalDevice.destroyShaderModule(fragShaderModule);
}

void Renderer::CreateComputePipeline() {
    // Set up programmable shaders
    vk::ShaderModule computeShaderModule = ShaderModule::Create("shaders/compute.comp.spv", logicalDevice);

    vk::PipelineShaderStageCreateInfo computeShaderStageInfo;
    computeShaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    computeShaderStageInfo.setModule(computeShaderModule);
    computeShaderStageInfo.setPName("main");
   
    // Add the compute descriptor set layout you create to this list
    std::array<vk::DescriptorSetLayout, 3> descriptorSetLayouts = { cameraDescriptorSetLayout, timeDescriptorSetLayout, computeDescriptorSetLayout };

    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayoutCount(static_cast<uint32_t>(descriptorSetLayouts.size()));
    pipelineLayoutInfo.setPSetLayouts(descriptorSetLayouts.data());
    pipelineLayoutInfo.setPushConstantRangeCount(0);
    pipelineLayoutInfo.setPushConstantRanges(0);
    
    try {
        computePipelineLayout = logicalDevice.createPipelineLayout(pipelineLayoutInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create compute pipeline layout");
    }

    // Create compute pipeline
    vk::ComputePipelineCreateInfo pipelineInfo;
    pipelineInfo.setStage(computeShaderStageInfo);
    pipelineInfo.setLayout(computePipelineLayout);
    pipelineInfo.setPNext(nullptr);
    // pipelineInfo.setFlags(vk::PipelineCreateFlags(0));
    pipelineInfo.setBasePipelineHandle(nullptr);
    pipelineInfo.setBasePipelineIndex(-1);
    
    try {
        computePipeline = (vk::Pipeline)logicalDevice.createComputePipeline(nullptr, pipelineInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    // No need for shader modules anymore
    vkDestroyShaderModule(logicalDevice, computeShaderModule, nullptr);
}

void Renderer::CreateFrameResources() {
    imageViews.resize(swapChain->GetCount());

    for (uint32_t i = 0; i < swapChain->GetCount(); i++) {
        // --- Create an image view for each swap chain image ---
        vk::ImageViewCreateInfo createInfo;
        createInfo.setImage(swapChain->GetVkImage(i));
      
        // Specify how the image data should be interpreted
        createInfo.setViewType(vk::ImageViewType::e2D);
        createInfo.setFormat(swapChain->GetVkImageFormat());

        // Specify color channel mappings (can be used for swizzling)
        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;

        // Describe the image's purpose and which part of the image should be accessed
        createInfo.subresourceRange.aspectMask = vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor);
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        // Create the image view
        try {
            imageViews[i] = logicalDevice.createImageView(createInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create image views");
        }
    }

    vk::Format depthFormat = device->GetInstance()->GetSupportedFormat({ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal, vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eDepthStencilAttachment));
    // CREATE DEPTH IMAGE
    Image::Create(device,
        swapChain->GetVkExtent().width,
        swapChain->GetVkExtent().height,
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlags(vk::ImageUsageFlagBits::eDepthStencilAttachment),
        vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal),
        depthImage,
        depthImageMemory
    );

    depthImageView = Image::CreateView(device, depthImage, depthFormat, vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth));
    
    // Transition the image for use as depth-stencil
    Image::TransitionLayout(device, graphicsCommandPool, depthImage, depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    
    // CREATE FRAMEBUFFERS
    framebuffers.resize(swapChain->GetCount());
    for (size_t i = 0; i < swapChain->GetCount(); i++) {
        std::vector<vk::ImageView> attachments = {
            imageViews[i],
            depthImageView
        };

        vk::FramebufferCreateInfo framebufferInfo;
        framebufferInfo.setRenderPass(renderPass);
        framebufferInfo.setAttachmentCount(static_cast<uint32_t>(attachments.size()));
        framebufferInfo.setPAttachments(attachments.data());
        framebufferInfo.setWidth(swapChain->GetVkExtent().width);
        framebufferInfo.setHeight(swapChain->GetVkExtent().height);
        framebufferInfo.setLayers(1);
        
        try {
            framebuffers[i] = logicalDevice.createFramebuffer(framebufferInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create framebuffer");
        }
    }
}

void Renderer::DestroyFrameResources() {
    for (size_t i = 0; i < imageViews.size(); i++) {
        logicalDevice.destroyImageView(imageViews[i]);
    }

    logicalDevice.destroyImageView(depthImageView);
    logicalDevice.freeMemory(depthImageMemory);
    logicalDevice.destroyImage(depthImage);
    
    for (size_t i = 0; i < framebuffers.size(); i++) {
        logicalDevice.destroyFramebuffer(framebuffers[i]);
    }
}

void Renderer::RecreateFrameResources() {
    logicalDevice.destroyPipeline(graphicsPipeline);
    logicalDevice.destroyPipeline(grassPipeline);
    logicalDevice.destroyPipelineLayout(graphicsPipelineLayout);
    logicalDevice.destroyPipelineLayout(grassPipelineLayout);
    logicalDevice.freeCommandBuffers(graphicsCommandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
  
    DestroyFrameResources();
    CreateFrameResources();
    CreateGraphicsPipeline();
    CreateGrassPipeline();
    RecordCommandBuffers();
}

void Renderer::RecordComputeCommandBuffer() {
    // Specify the command pool and number of buffers to allocate
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(computeCommandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);
 
    try {
        logicalDevice.allocateCommandBuffers(&allocInfo, &computeCommandBuffer);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate record command buffers");
    }

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse));
    beginInfo.setPInheritanceInfo(nullptr);
   
    // ~ Start recording ~
    try {
        computeCommandBuffer.begin(beginInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to begin recording compute command buffer");
    }

    // Bind to the compute pipeline
    computeCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);

    // Bind camera descriptor set
    computeCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, 1, &cameraDescriptorSet, 0, nullptr);

    // Bind descriptor set for time uniforms
    computeCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 1, 1, &timeDescriptorSet, 0, nullptr);

    // For each group of blades bind its descriptor set and dispatch
    for (int i = 0; i < computeDescriptorSets.size(); i++) {
        computeCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 2, 1, &computeDescriptorSets[i], 0, nullptr);
        computeCommandBuffer.dispatch(NUM_BLADES / WORKGROUP_SIZE + 1, 1, 1);
    }

    // ~ End recording ~
    try {
        computeCommandBuffer.end();
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to end record compute command buffer");
    }
}

void Renderer::RecordCommandBuffers() {
    commandBuffers.resize(swapChain->GetCount());

    // Specify the command pool and number of buffers to allocate
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(graphicsCommandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(static_cast<uint32_t>(commandBuffers.size()));  // commandBuffers has been resized to swapChain->GetCount()
    
    try {
        commandBuffers = logicalDevice.allocateCommandBuffers(allocInfo);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to allocate command buffers");
    }

    // Start command buffer recording
    for (size_t i = 0; i < commandBuffers.size(); i++) {
        vk::CommandBufferBeginInfo beginInfo;
        beginInfo.setFlags(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse));
        beginInfo.setPInheritanceInfo(nullptr);
       
        // Start recording 
        try {
            commandBuffers[i].begin(beginInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("Failed to begin recording command buffer");
        }

        // Begin the render pass
        vk::RenderPassBeginInfo renderPassInfo;
        renderPassInfo.setRenderPass(renderPass);
        renderPassInfo.setFramebuffer(framebuffers[i]);
        renderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
        renderPassInfo.renderArea.extent = swapChain->GetVkExtent();

        std::array<vk::ClearValue, 2> clearValues = {};
        clearValues[0].setColor(vk::ClearColorValue(std::array<float, 4>{0.4f, 0.78f, 1.0f, 1.0f}));
        clearValues[1].setDepthStencil({ 1.0f, 0 });
        renderPassInfo.setClearValueCount(static_cast<uint32_t>(clearValues.size()));
        renderPassInfo.setPClearValues(clearValues.data());
         
        std::vector<vk::BufferMemoryBarrier> barriers(scene->GetBlades().size());
        for (uint32_t j = 0; j < barriers.size(); ++j) {
            barriers[j].setSrcAccessMask(vk::AccessFlags(vk::AccessFlagBits::eShaderWrite));
            barriers[j].setDstAccessMask(vk::AccessFlags(vk::AccessFlagBits::eIndirectCommandRead));
            barriers[j].setSrcQueueFamilyIndex(device->GetQueueIndex(QueueFlags::Compute));
            barriers[j].setDstQueueFamilyIndex(device->GetQueueIndex(QueueFlags::Graphics));
            barriers[j].setBuffer(scene->GetBlades()[j]->GetNumBladesBuffer());
            barriers[j].setOffset(0);
            barriers[j].setSize(sizeof(BladeDrawIndirect));
        }

        commandBuffers[i].pipelineBarrier(vk::PipelineStageFlags(vk::PipelineStageFlagBits::eComputeShader), 
            vk::PipelineStageFlags(vk::PipelineStageFlagBits::eDrawIndirect),
            vk::DependencyFlags(0),
            0, nullptr, barriers.size(), barriers.data(), 0, nullptr);

        // Bind the camera descriptor set. This is set 0 in all pipelines so it will be inherited
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipelineLayout, 0, 1, &cameraDescriptorSet, 0, nullptr);

        commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

        // Bind the graphics pipeline
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

        for (uint32_t j = 0; j < scene->GetModels().size(); ++j) {
            // Bind the vertex and index buffers
            std::array<vk::Buffer, 1> vertexBuffers = { scene->GetModels()[j]->getVertexBuffer() };
            std::array<vk::DeviceSize, 1> offsets = { 0 };
            commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());

            commandBuffers[i].bindIndexBuffer(scene->GetModels()[j]->getIndexBuffer(), 0, vk::IndexType::eUint32);

            // Bind the descriptor set for each model
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipelineLayout, 1, 1, &modelDescriptorSets[j], 0, nullptr);

            // Draw
            std::vector<uint32_t> indices = scene->GetModels()[j]->getIndices();
            commandBuffers[i].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        }

        // Bind the grass pipeline
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, grassPipeline);

        for (uint32_t j = 0; j < scene->GetBlades().size(); ++j) {
            std::array<vk::Buffer, 1> vertexBuffers = { scene->GetBlades()[j]->GetCulledBladesBuffer() };
            std::array<vk::DeviceSize, 1> offsets = { 0 };
            commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());

            // Bind the descriptor set for each grass blades model
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, grassPipelineLayout, 1, 1, &grassDescriptorSets[j], 0, nullptr);
            // Draw
            commandBuffers[i].drawIndirect(scene->GetBlades()[j]->GetNumBladesBuffer(), 0, 1, sizeof(BladeDrawIndirect));
        }

        // End render pass
        commandBuffers[i].endRenderPass();

        // ~ End recording ~
        try {
            commandBuffers[i].end();
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("Failed to end recording command buffer");
        }
    }
}

void Renderer::Frame() {
    vk::SubmitInfo computeSubmitInfo;
    computeSubmitInfo.setCommandBufferCount(1);
    computeSubmitInfo.setPCommandBuffers(&computeCommandBuffer);
   
    try {
        device->GetQueue(QueueFlags::Compute).submit(computeSubmitInfo, nullptr);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }

    if (!swapChain->Acquire()) {
        RecreateFrameResources();
        return;
    }

    // Submit the command buffer
    vk::SubmitInfo submitDrawInfo;
    
    std::array<vk::Semaphore, 1> waitSemaphores = { swapChain->GetImageAvailableVkSemaphore() };
    std::array<vk::PipelineStageFlags, 1> waitStages = { vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput) };
    submitDrawInfo.setWaitSemaphoreCount(1);
    submitDrawInfo.setPWaitSemaphores(waitSemaphores.data());
    submitDrawInfo.setPWaitDstStageMask(waitStages.data());

    submitDrawInfo.setCommandBufferCount(1);
    submitDrawInfo.setPCommandBuffers(&commandBuffers[swapChain->GetIndex()]);

    std::array<vk::Semaphore, 1> signalSemaphores = { swapChain->GetRenderFinishedVkSemaphore() };
    submitDrawInfo.setSignalSemaphoreCount(1);
    submitDrawInfo.setPSignalSemaphores(signalSemaphores.data());

    try {
        device->GetQueue(QueueFlags::Graphics).submit(submitDrawInfo, nullptr);
    }
    catch (vk::SystemError err) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }

    if (!swapChain->Present()) {
        RecreateFrameResources();
    }
}

Renderer::~Renderer() {
    logicalDevice.waitIdle();

    // TODO: destroy any resources you created

    logicalDevice.freeCommandBuffers(graphicsCommandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    logicalDevice.freeCommandBuffers(computeCommandPool, 1, &computeCommandBuffer);
    
    logicalDevice.destroyPipeline(graphicsPipeline);
    logicalDevice.destroyPipeline(grassPipeline);
    logicalDevice.destroyPipeline(computePipeline);

    logicalDevice.destroyPipelineLayout(graphicsPipelineLayout);
    logicalDevice.destroyPipelineLayout(grassPipelineLayout);
    logicalDevice.destroyPipelineLayout(computePipelineLayout);

    logicalDevice.destroyDescriptorSetLayout(cameraDescriptorSetLayout);
    logicalDevice.destroyDescriptorSetLayout(modelDescriptorSetLayout);
    logicalDevice.destroyDescriptorSetLayout(timeDescriptorSetLayout);
    logicalDevice.destroyDescriptorSetLayout(computeDescriptorSetLayout);
    
    logicalDevice.destroyDescriptorPool(descriptorPool);
   
    logicalDevice.destroyRenderPass(renderPass);
    
    DestroyFrameResources();
    logicalDevice.destroyCommandPool(computeCommandPool);
    logicalDevice.destroyCommandPool(graphicsCommandPool);
}

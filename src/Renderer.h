#pragma once

#include "Device.h"
#include "SwapChain.h"
#include "Scene.h"
#include "Camera.h"

class Renderer {
public:
    Renderer() = delete;
    Renderer(Device* device, SwapChain* swapChain, Scene* scene, Camera* camera);
    ~Renderer();

    void CreateCommandPools();

    void CreateRenderPass();

    void CreateCameraDescriptorSetLayout();
    void CreateModelDescriptorSetLayout();
    void CreateTimeDescriptorSetLayout();
    void CreateComputeDescriptorSetLayout();

    void CreateDescriptorPool();

    void CreateCameraDescriptorSet();
    void CreateModelDescriptorSets();
    void CreateGrassDescriptorSets();
    void CreateTimeDescriptorSet();
    void CreateComputeDescriptorSets();

    void CreateGraphicsPipeline();
    void CreateGrassPipeline();
    void CreateComputePipeline();

    void CreateFrameResources();
    void DestroyFrameResources();
    void RecreateFrameResources();

    void RecordCommandBuffers();
    void RecordComputeCommandBuffer();

    void Frame();

private:
    Device* device;
    vk::Device logicalDevice;
    SwapChain* swapChain;
    Scene* scene;
    Camera* camera;

    vk::CommandPool graphicsCommandPool;
    vk::CommandPool computeCommandPool;

    vk::RenderPass renderPass;

    vk::DescriptorSetLayout cameraDescriptorSetLayout;
    vk::DescriptorSetLayout modelDescriptorSetLayout;
    vk::DescriptorSetLayout timeDescriptorSetLayout;
    vk::DescriptorSetLayout computeDescriptorSetLayout;
    
    vk::DescriptorPool descriptorPool;

    vk::DescriptorSet cameraDescriptorSet;
    std::vector<vk::DescriptorSet> modelDescriptorSets;
    vk::DescriptorSet timeDescriptorSet;
    std::vector<vk::DescriptorSet> computeDescriptorSets;
    std::vector<vk::DescriptorSet> grassDescriptorSets;

    vk::PipelineLayout graphicsPipelineLayout;
    vk::PipelineLayout grassPipelineLayout;
    vk::PipelineLayout computePipelineLayout;

    vk::Pipeline graphicsPipeline;
    vk::Pipeline grassPipeline;
    vk::Pipeline computePipeline;

    std::vector<vk::ImageView> imageViews;
    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;
    std::vector<vk::Framebuffer> framebuffers;

    std::vector<vk::CommandBuffer> commandBuffers;
    vk::CommandBuffer computeCommandBuffer;
};

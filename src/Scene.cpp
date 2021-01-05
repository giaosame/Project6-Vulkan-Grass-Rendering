#include "Scene.h"
#include "BufferUtils.h"

Scene::Scene(Device* device) 
    : device(device)
{
    BufferUtils::CreateBuffer(device, sizeof(Time), vk::BufferUsageFlags(vk::BufferUsageFlagBits::eUniformBuffer), vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent), timeBuffer, timeBufferMemory);
    mappedData = device->GetLogicalDevice().mapMemory(timeBufferMemory, 0, sizeof(Time));
    memcpy(mappedData, &time, sizeof(Time));
}

const std::vector<Model*>& Scene::GetModels() const {
    return models;
}

const std::vector<Blades*>& Scene::GetBlades() const {
    return blades;
}

void Scene::AddModel(Model* model) {
    models.push_back(model);
}

void Scene::AddBlades(Blades* blades) {
    this->blades.push_back(blades);
}

void Scene::UpdateTime() {
    high_resolution_clock::time_point currentTime = high_resolution_clock::now();
    duration<float> nextDeltaTime = duration_cast<duration<float>>(currentTime - startTime);
    startTime = currentTime;

    time.deltaTime = nextDeltaTime.count();
    time.totalTime += time.deltaTime;

    memcpy(mappedData, &time, sizeof(Time));
}

vk::Buffer Scene::GetTimeBuffer() const {
    return timeBuffer;
}

Scene::~Scene() {
    device->GetLogicalDevice().unmapMemory(timeBufferMemory);
    device->GetLogicalDevice().destroyBuffer(timeBuffer);
    device->GetLogicalDevice().freeMemory(timeBufferMemory);
}

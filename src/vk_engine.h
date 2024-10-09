// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "vector"
#include "deque"
#include "functional"
#include "vk_mem_alloc.h"
#include "vk_mesh.h"
#include "glm/glm.hpp"
#include "unordered_map"

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()> &&function) {
        deletors.push_back(function);
    }

    void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();
        }
        deletors.clear();
    }
};

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
};

struct Material {
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct RenderObject {
    Mesh *mesh;
    Material *material;
    glm::mat4 transformMatrix;
};

struct GPUCameraData{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
};

struct FrameData{
   VkSemaphore _presentSemaphore, _renderSemaphore;
   VkFence _renderFence;

   VkCommandPool _commandPool;
   VkCommandBuffer _mainCommandBuffer;

   AllocatedBuffer _cameraBuffer;
   VkDescriptorSet _globalDescriptor;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};

    VkDescriptorSetLayout _globalSetLayout;
    VkDescriptorPool _descriptorPool;

    FrameData _frames[FRAME_OVERLAP];

    DeletionQueue _mainDeletionQueue;
    VmaAllocator _allocator;

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;
    VkSurfaceKHR _surface;

    VkPhysicalDeviceProperties _gpuProperties;

    std::vector<RenderObject> _renderables;
    std::unordered_map<std::string, Material> _materials;
    std::unordered_map<std::string, Mesh> _meshes;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    VkRenderPass _renderPass;

    // Depth Image
    VkImageView _depthImageView;
    AllocatedImage _depthImage;
    VkFormat _depthFormat;
    //End Depth Image

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    VkExtent2D _windowExtent{1700, 900};

    std::vector<VkFramebuffer> _framebuffers;

    struct SDL_Window *_window{nullptr};

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw();

    //run main loop
    void run();

    Material *create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name);

    Material *get_material(const std::string &name);

    Mesh *get_mesh(const std::string &name);

    void draw_objects(VkCommandBuffer cmd, RenderObject *first, int count);

    FrameData &get_current_frame();

private:

    void init_vulkan();

    void init_swapchain();

    void init_commands();

    void init_default_renderpass();

    void init_framebuffers();

    void init_sync_structures();

    void init_pipelines();

    void init_scene();

    void init_descriptors();

    bool load_shader_module(const char *filePath, VkShaderModule *outShaderModule);

    void load_meshes();

    void upload_mesh(Mesh &mesh);

    AllocatedBuffer create_buffer(size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) const;

    void write_buffer(void *pData, VkDeviceSize size, VmaAllocation allocation);
};

class PipelineBuilder {

public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkViewport _viewport;
    VkRect2D _scissor;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineDepthStencilStateCreateInfo _depthStencil;
    VkPipelineLayout _pipelineLayout;

    VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

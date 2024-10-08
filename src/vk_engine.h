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

class VulkanEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};

    DeletionQueue _mainDeletionQueue;
    VmaAllocator _allocator;

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;
    VkSurfaceKHR _surface;

    std::vector<RenderObject> _renderables;
    std::unordered_map<std::string, Material> _materials;
    std::unordered_map<std::string, Mesh> _meshes;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkRenderPass _renderPass;
    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

    VkPipeline _meshPipeline;
    VkPipelineLayout _meshPipelineLayout;
    Mesh _triangleMesh;
    Mesh _monkeyMesh;

    // Depth Image
    VkImageView _depthImageView;
    AllocatedImage _depthImage;
    VkFormat _depthFormat;
    //End Depth Image

    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

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

    Material* get_material(const std::string &name);

    Mesh* get_mesh(const std::string &name);

    void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

private:

    void init_vulkan();

    void init_swapchain();

    void init_commands();

    void init_default_renderpass();

    void init_framebuffers();

    void init_sync_structures();

    void init_pipelines();

    void init_scene();

    bool load_shader_module(const char *filePath, VkShaderModule *outShaderModule);

    void load_meshes();

    void upload_mesh(Mesh &mesh);

    void
    create_buffer(VkDeviceSize size, void *pData, VkBufferUsageFlags usage, VkBuffer *pBuffer,
                  VmaAllocation *pAllocation);
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

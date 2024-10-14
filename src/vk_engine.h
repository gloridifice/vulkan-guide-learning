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
#include "imgui_impl_sdl.h"

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
    VkDescriptorSet textureSet{VK_NULL_HANDLE};
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct Texture {
    AllocatedImage image;
    VkImageView imageView;
};

struct RenderObject {
    Mesh *mesh;
    Material *material;
    glm::mat4 transformMatrix;
};

struct GPUCameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
};

struct GPUSceneData {
    glm::vec4 fogColor;
    glm::vec4 fogDistances;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;
};

struct FrameData {
    VkSemaphore presentSemaphore, renderSemaphore;
    VkFence renderFence;

    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    AllocatedBuffer cameraBuffer;
    VkDescriptorSet globalDescriptor;

    AllocatedBuffer objectBuffer;
    VkDescriptorSet objectDescriptor;
};

struct GPUObjectData {
    glm::mat4 modelMatrix;
};

struct UploadContext {
    VkFence uploadFence;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
};

class Camera {
public:
    glm::vec3 position = glm::vec3();
    float pitch = 90.f;
    float yaw = 0.f;
    float fovy = glm::radians(70.f);
    float aspect = 1700.f / 900.f;
    float zNear = 0.1f;
    float zFar = 200.f;

    GPUCameraData gpu_data();
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};

    Camera* _camera = new Camera();

    VkDescriptorSetLayout _singleTextureSetLayout;
    UploadContext _uploadContext;

    GPUSceneData _sceneParameters;
    AllocatedBuffer _sceneParameterBuffer;

    VkDescriptorSetLayout _globalSetLayout;
    VkDescriptorSetLayout _objectSetLayout;
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
    std::unordered_map<std::string, Texture> _loadedTextures;

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

    void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

    AllocatedBuffer create_buffer(size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) const;

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

    void load_images();

    void upload_mesh(Mesh &mesh);

    void write_buffer(void *pData, VkDeviceSize size, VmaAllocation allocation, size_t offset = 0);

    size_t pad_uniform_buffer_size(size_t originalSize);

    VkShaderModule new_shader_module_from_loading(const std::string &filePath);

    void copy_buffer_cpu2gpu(size_t bufferSize, void *pData, VkBuffer dstBuffer, VkDeviceSize dstOffset = 0,
                             VkDeviceSize srcOffset = 0);

    void handle_input(SDL_Event &e);

    void update();
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

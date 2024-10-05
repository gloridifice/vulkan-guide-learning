// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "vector"
#include "deque"
#include "functional"

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function){
        deletors.push_back(function);
    }

    void flush(){
        for (auto it = deletors.rbegin(); it != deletors.rend(); it ++) {
            (*it)();
        }
        deletors.clear();
    }
};

class VulkanEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};

    DeletionQueue _mainDeletionQueue;

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;
    VkSurfaceKHR _surface;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkRenderPass _renderPass;
    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

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

private:

    void init_vulkan();

    void init_swapchain();

    void init_commands();

    void init_default_renderpass();

    void init_framebuffers();

    void init_sync_structures();

    void init_pipelines();

    bool load_shader_module(const char *filePath, VkShaderModule *outShaderModule);
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
    VkPipelineLayout _pipelineLayout;

    VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

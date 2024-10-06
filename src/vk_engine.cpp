
#define VMA_IMPLEMENTATION

#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include <fstream>
#include <glm/gtx/transform.hpp>

#include "cerrno"
#include "iostream"

#include "VkBootstrap.h"

#define VK_CHECK(f)                                                            \
{                                                                              \
    VkResult res = (f);                                                        \
    if (res != VK_SUCCESS) {                                                   \
        std::cerr << "Fatal : VkResult is \"" << res << "\" in " << __FILE__    \
                  << " at line " << __LINE__ << std::endl;                     \
        assert(res == VK_SUCCESS);                                             \
    }                                                                          \
}

void VulkanEngine::init() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags) (SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow(
            "Vulkan Engine",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            _windowExtent.width,
            _windowExtent.height,
            window_flags
    );

    init_vulkan();
    init_swapchain();
    init_commands();
    init_default_renderpass();
    init_framebuffers();
    init_sync_structures();
    init_pipelines();

    load_meshes();

    //everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(_device);
        vkWaitForFences(_device, 1, &_renderFence, true, 1000000000);

        _mainDeletionQueue.flush();

        vmaDestroyAllocator(_allocator);
        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }
}

void VulkanEngine::draw() {
    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &_renderFence));

    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapchainImageIndex))
    VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0))

    //naming it cmd for shorter writing
    VkCommandBuffer cmd = _mainCommandBuffer;

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    VkClearValue clearValue;
    float flash = abs(sin(_frameNumber / 120.0f));
    clearValue.color = {{0.0f, 0.0f, flash, 1.0f}};

    VkRenderPassBeginInfo rpInfo = vkinit::render_pass_begin_info(_renderPass, _windowExtent,
                                                                  _framebuffers[swapchainImageIndex], &clearValue);

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Render Code Start ---

    glm::vec3 camPos = { 0.f,0.f,-2.f };

    glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
    //camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
    projection[1][1] *= -1;
    //model rotation
    glm::mat4 model = glm::rotate(glm::mat4{ 1.0f }, glm::radians(_frameNumber * 0.4f), glm::vec3(0, 1, 0));

    //calculate final mesh matrix
    glm::mat4 mesh_matrix = projection * view * model;

    MeshPushConstants constants;
    constants.render_matrix = mesh_matrix;

    //upload the matrix to the GPU via push constants
    vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &_triangleMesh._vertexBuffer._buffer, &offset);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Render Code End ---

    vkCmdEndRenderPass(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = nullptr;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    submit.pWaitDstStageMask = &waitStage;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &_presentSemaphore;

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &_renderSemaphore;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;

    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &_renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    //increase the number of frames drawn
    _frameNumber++;
}

void VulkanEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    //main loop
    while (!bQuit) {
        //Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            //close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT) bQuit = true;
        }

        draw();
    }
}

void VulkanEngine::init_vulkan() {
    // Instance & Messenger
    vkb::InstanceBuilder builder;
    auto instance_result = builder.set_app_name("Hello Vulkan Guide")
            .request_validation_layers(true)
            .use_default_debug_messenger()
            .enable_extension("VK_EXT_metal_surface")
            .enable_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
            .require_api_version(1, 1, 0)
            .build();

    vkb::Instance vkb_instance = instance_result.value();

    _instance = vkb_instance.instance;
    _debug_messenger = vkb_instance.debug_messenger;

    // Surface
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    vkb::PhysicalDeviceSelector selector{vkb_instance};
    vkb::PhysicalDevice physicalDevice = selector
            .set_minimum_version(1, 1)
            .set_surface(_surface)
            .add_desired_extension("VK_KHR_portability_subset")
            .select()
            .value();

    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // init the memory allocator
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.physicalDevice = _chosenGPU;
    allocatorCreateInfo.device = _device;
    allocatorCreateInfo.instance = _instance;
    vmaCreateAllocator(&allocatorCreateInfo, &_allocator);
}

void VulkanEngine::init_swapchain() {
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    vkb::Swapchain vkbSwapchain = swapchainBuilder
            .use_default_format_selection()
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(_windowExtent.width, _windowExtent.height)
            .build()
            .value();

    //store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

    _swapchainImageFormat = vkbSwapchain.image_format;

    _mainDeletionQueue.push_function([=]() {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    });
}

void VulkanEngine::init_commands() {
    //create a command pool for commands submitted to the graphics queue.
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily,
                                                                               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

    //allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _commandPool, nullptr);
    });
}

void VulkanEngine::init_sync_structures() {
    // Fence
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));
    _mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_device, _renderFence, nullptr);
    });

    // for the semaphores we don't need any flags
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info(0);
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));
    _mainDeletionQueue.push_function([=]() {
        vkDestroySemaphore(_device, _presentSemaphore, nullptr);
        vkDestroySemaphore(_device, _renderSemaphore, nullptr);
    });
}

void VulkanEngine::init_default_renderpass() {
    VkAttachmentDescription color_attachment = {};
    color_attachment.format = _swapchainImageFormat;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    //we don't care about stencil
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    //attachment number will index into the pAttachments array in the parent renderpass itself
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    //we are going to create 1 subpass, which is the minimum you can do
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

    //connect the color attachment to the info
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    //connect the subpass to the info
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;

    VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyRenderPass(_device, _renderPass, nullptr);
    });
}

void VulkanEngine::init_framebuffers() {
    //create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
    VkFramebufferCreateInfo fb_info = {};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;

    fb_info.renderPass = _renderPass;
    fb_info.attachmentCount = 1;
    fb_info.width = _windowExtent.width;
    fb_info.height = _windowExtent.height;
    fb_info.layers = 1;

    //grab how many images we have in the swapchain
    const uint32_t swapchain_imagecount = _swapchainImages.size();
    _framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

    //create framebuffers for each of the swapchain image views
    for (int i = 0; i < swapchain_imagecount; i++) {
        fb_info.pAttachments = &_swapchainImageViews[i];
        VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
    }
}

void VulkanEngine::init_pipelines() {
    VkShaderModule triangleFragShader;
    if (!load_shader_module("../shaders/triangle.frag.spv", &triangleFragShader)) {
        std::cout << "Error when building the triangle fragment shader module" << std::endl;
    } else {
        std::cout << "Triangle fragment shader successfully loaded" << std::endl;
    }
    VkShaderModule triangleVertexShader;
    if (!load_shader_module("../shaders/triangle.vert.spv", &triangleVertexShader)) {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    } else {
        std::cout << "Triangle vertex shader successfully loaded" << std::endl;
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

    //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;

    pipelineBuilder._shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triangleVertexShader));
    pipelineBuilder._shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));
    pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
    pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder._viewport.x = 0.0f;
    pipelineBuilder._viewport.y = 0.0f;
    pipelineBuilder._viewport.width = (float) _windowExtent.width;
    pipelineBuilder._viewport.height = (float) _windowExtent.height;
    pipelineBuilder._viewport.minDepth = 0.0f;
    pipelineBuilder._viewport.maxDepth = 1.0f;
    pipelineBuilder._scissor.offset = {0, 0};
    pipelineBuilder._scissor.extent = _windowExtent;
    pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
    pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();
    pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();
    pipelineBuilder._pipelineLayout = _trianglePipelineLayout;

    _trianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

    // Mesh Pipeline

    VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();

    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = sizeof(MeshPushConstants);
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
    mesh_pipeline_layout_info.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr, &_meshPipelineLayout));

    auto description = Vertex::get_vertex_description();
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = description.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = description.attributes.size();

    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = description.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = description.bindings.size();
    pipelineBuilder._shaderStages.clear();

    VkShaderModule meshVertShader;
    if (!load_shader_module("../shaders/tri_mesh.vert.spv", &meshVertShader)) {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    } else {
        std::cout << "Red Triangle vertex shader successfully loaded" << std::endl;
    }

    pipelineBuilder._pipelineLayout = _meshPipelineLayout;

    pipelineBuilder._shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

    _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);
    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);

        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    });
}

bool VulkanEngine::load_shader_module(const char *filePath, VkShaderModule *outShaderModule) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    size_t fileSize = (size_t) file.tellg();

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char *) buffer.data(), fileSize);
    file.close();

    auto createInfo = vkinit::shader_module_create_info(buffer.size() * sizeof(uint32_t), buffer.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass) {
    //make viewport state from our stored viewport and scissor.
    //at the moment we won't support multiple viewports or scissors
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.pViewports = &_viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &_scissor;

    //setup dummy color blending. We aren't using transparent objects yet
    //the blending is just "no blend", but we do write to the color attachment
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;

    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &_colorBlendAttachment;

    //build the actual pipeline
    //we now use all of the info structs we have been writing into into this one to create the pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;

    pipelineInfo.stageCount = _shaderStages.size();
    pipelineInfo.pStages = _shaderStages.data();
    pipelineInfo.pVertexInputState = &_vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &_inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &_rasterizer;
    pipelineInfo.pMultisampleState = &_multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = _pipelineLayout;
    pipelineInfo.renderPass = pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    //it's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
        std::cout << "failed to create pipeline\n";
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    } else {
        return newPipeline;
    }
}

void VulkanEngine::load_meshes() {
    //make the array 3 vertices long
    _triangleMesh._vertices.resize(3);

    //vertex positions
    _triangleMesh._vertices[0].position = {1.f, 1.f, 0.0f};
    _triangleMesh._vertices[1].position = {-1.f, 0.5f, 0.0f};
    _triangleMesh._vertices[2].position = {0.f, -1.f, 0.0f};

    //vertex colors, all green
    _triangleMesh._vertices[0].color = {0.f, 1.f, 0.0f}; //pure green
    _triangleMesh._vertices[1].color = {0.f, 1.f, 0.0f}; //pure green
    _triangleMesh._vertices[2].color = {0.f, 1.f, 0.0f}; //pure green

    //we don't care about the vertex normals

    upload_mesh(_triangleMesh);
}

void VulkanEngine::upload_mesh(Mesh &mesh) {
    //allocate vertex buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    //this is the total size, in bytes, of the buffer we are allocating
    bufferInfo.size = mesh._vertices.size() * sizeof(Vertex);
    //this buffer is going to be used as a Vertex Buffer
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;


    //let the VMA library know that this data should be writeable by CPU, but also readable by GPU
    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    //allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
                             &mesh._vertexBuffer._buffer,
                             &mesh._vertexBuffer._allocation,
                             nullptr));

    //add the destruction of triangle mesh buffer to the deletion queue
    _mainDeletionQueue.push_function([=]() {

        vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
    });

    void *data;
    vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);

    memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));

    vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);

}


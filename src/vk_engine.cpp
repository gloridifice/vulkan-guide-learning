
#define VMA_IMPLEMENTATION
#define GLM_ENABLE_EXPERIMENTAL

#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include "input/vk_input.h"
#include <vk_initializers.h>
#include "vk_textures.h"
#include <glm/gtx/transform.hpp>
#include "input/Input.h"

#include <fstream>
#include "cerrno"
#include "iostream"
#include "array"
#include "optional"

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
    init_descriptors();
    init_pipelines();

    load_images();
    load_meshes();
    init_scene();

    //everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(_device);
        vkWaitForFences(_device, 1, &get_current_frame().renderFence, true, 1000000000);

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
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame().renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame().renderFence));

    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame().presentSemaphore, nullptr,
                                   &swapchainImageIndex))
    VK_CHECK(vkResetCommandBuffer(get_current_frame().mainCommandBuffer, 0))

    //naming it cmd for shorter writing
    VkCommandBuffer cmd = get_current_frame().mainCommandBuffer;

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    VkClearValue clearValue;
    clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkClearValue depthClearValue;
    depthClearValue.depthStencil.depth = 1.f;

    std::array<VkClearValue, 2> clearValues = {clearValue, depthClearValue};

    VkRenderPassBeginInfo rpInfo = vkinit::render_pass_begin_info(_renderPass, _windowExtent,
                                                                  _framebuffers[swapchainImageIndex],
                                                                  clearValues.data());
    rpInfo.clearValueCount = clearValues.size();

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Render Code Start ---

    draw_objects(cmd, _renderables.data(), _renderables.size());

    // Render Code End ---

    vkCmdEndRenderPass(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = vkinit::submit_info(&cmd);
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    submit.pWaitDstStageMask = &waitStage;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &get_current_frame().presentSemaphore;

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &get_current_frame().renderSemaphore;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, get_current_frame().renderFence));

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;

    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &get_current_frame().renderSemaphore;
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
            else handle_input(e);
        }

        update();
        draw();
    }
}

void VulkanEngine::update() {
    glm::vec3 movement{};
    if (vkinput::get_key(SDLK_w)) movement += glm::vec3{0.f, 0.f, 1.f};
    if (vkinput::get_key(SDLK_s)) movement += glm::vec3{0.f, 0.f, -1.f};
    if (vkinput::get_key(SDLK_a)) movement += glm::vec3{1.f, 0.f, 0.f};
    if (vkinput::get_key(SDLK_d)) movement += glm::vec3{-1.f, 0.f, 0.f};
    if (vkinput::get_key(SDLK_SPACE)) movement += glm::vec3{0.f, 1.f, 0.f};
    if (vkinput::get_key(SDLK_LSHIFT)) movement += glm::vec3{0.f, -1.f, 0.f};

    if (movement != glm::vec3()) {
        _camera->position += glm::normalize(movement);
    }
}

void VulkanEngine::handle_input(SDL_Event &e) {
    if (e.type == SDL_KEYDOWN) {
        SDL_Keycode sym = e.key.keysym.sym;
        Input::INSTANCE->_keys.insert_or_assign(sym, true);
    } else if (e.type == SDL_KEYUP) {
        auto sym = e.key.keysym.sym;
        Input::INSTANCE->_keys.insert_or_assign(sym, false);
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
    VkPhysicalDeviceShaderDrawParametersFeatures shaderDrawParametersFeatures{};
    shaderDrawParametersFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
    shaderDrawParametersFeatures.pNext = nullptr;
    shaderDrawParametersFeatures.shaderDrawParameters = VK_TRUE;

    vkb::Device vkbDevice = deviceBuilder.add_pNext(&shaderDrawParametersFeatures).build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    _gpuProperties = physicalDevice.properties;

    std::cout << "The GPU has a minimum buffer alignment of " << _gpuProperties.limits.minUniformBufferOffsetAlignment
              << std::endl;

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

    VkExtent3D depthImageExtent = {
            _windowExtent.width,
            _windowExtent.height,
            1
    };

    // TODO refactor creating image
    _depthFormat = VK_FORMAT_D32_SFLOAT;
    auto dImageInfo = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                                depthImageExtent);
    VmaAllocationCreateInfo dImageAllocInfo = {};
    dImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dImageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vmaCreateImage(_allocator, &dImageInfo, &dImageAllocInfo, &_depthImage._image, &_depthImage._allocation, nullptr);
    auto dViewInfo = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dViewInfo, nullptr, &_depthImageView))

    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _depthImageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    });
}

void VulkanEngine::init_commands() {
    //create a command pool for commands submitted to the graphics queue.
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily,
                                                                               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        auto &frame = _frames[i];
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &frame.commandPool));

        //allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(frame.commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &frame.mainCommandBuffer));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyCommandPool(_device, frame.commandPool, nullptr);
        });
    }

    auto uploadPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
    VK_CHECK(vkCreateCommandPool(_device, &uploadPoolInfo, nullptr, &_uploadContext.commandPool));
    _mainDeletionQueue.push_function([=] {
        vkDestroyCommandPool(_device, _uploadContext.commandPool, nullptr);
    });
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext.commandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_uploadContext.commandBuffer));
}

void VulkanEngine::init_sync_structures() {
    // Create render fence and semaphores
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info(0);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        auto &frame = _frames[i];

        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &frame.renderFence));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &frame.presentSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &frame.renderSemaphore));
        _mainDeletionQueue.push_function([=]() {
            vkDestroyFence(_device, frame.renderFence, nullptr);
            vkDestroySemaphore(_device, frame.presentSemaphore, nullptr);
            vkDestroySemaphore(_device, frame.renderSemaphore, nullptr);
        });
    }

    // Create upload fence
    auto uploadFenceCreateInfo = vkinit::fence_create_info(0);
    VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext.uploadFence));
    _mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_device, _uploadContext.uploadFence, nullptr);
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

    // Depth Attachment
    VkAttachmentDescription depth_attachment = {};
    depth_attachment.flags = 0;
    depth_attachment.format = _depthFormat;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    //we are going to create 1 subpass, which is the minimum you can do
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    std::array<VkAttachmentDescription, 2> attachments = {color_attachment, depth_attachment};

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkSubpassDependency depth_dependency = {};
    depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    depth_dependency.dstSubpass = 0;
    depth_dependency.srcStageMask =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depth_dependency.srcAccessMask = 0;
    depth_dependency.dstStageMask =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkSubpassDependency, 2> dependencies = {dependency, depth_dependency};

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = attachments.size();
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = dependencies.size();
    render_pass_info.pDependencies = dependencies.data();

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
        std::array<VkImageView, 2> attachments{};
        attachments[0] = _swapchainImageViews[i];
        attachments[1] = _depthImageView;

        fb_info.pAttachments = attachments.data();
        fb_info.attachmentCount = attachments.size();
        VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
    }
}

VkShaderModule VulkanEngine::new_shader_module_from_loading(const std::string &filePath) {
    VkShaderModule ret;
    if (!load_shader_module(filePath.c_str(), &ret)) {
        std::cout << "Error when building the triangle fragment shader module" << std::endl;
    } else {
        std::cout << "Shader Loaded: " << filePath << std::endl;
    }
    return ret;
}

void VulkanEngine::init_pipelines() {
    VkShaderModule meshVertShader = new_shader_module_from_loading("../shaders/tri_mesh.vert.spv");
    VkShaderModule colorMeshFragShader = new_shader_module_from_loading("../shaders/default_lit.frag.spv");

    //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;

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
    pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // Mesh Pipeline
    VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = vkinit::pipeline_layout_create_info();

    std::array<VkDescriptorSetLayout, 3> setLayouts = {_globalSetLayout, _objectSetLayout, _singleTextureSetLayout};

    VkPushConstantRange pushConstant;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(MeshPushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    meshPipelineLayoutInfo.pPushConstantRanges = &pushConstant;
    meshPipelineLayoutInfo.pushConstantRangeCount = 1;

    meshPipelineLayoutInfo.setLayoutCount = setLayouts.size();
    meshPipelineLayoutInfo.pSetLayouts = setLayouts.data();

    VkPipelineLayout meshPipelineLayout;

    VK_CHECK(vkCreatePipelineLayout(_device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

    auto description = Vertex::get_vertex_description();
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = description.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = description.attributes.size();

    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = description.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = description.bindings.size();
    pipelineBuilder._shaderStages.clear();

    pipelineBuilder._pipelineLayout = meshPipelineLayout;

    pipelineBuilder._shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, colorMeshFragShader));

    auto meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

    create_material(meshPipeline, meshPipelineLayout, "default_mesh");

    vkDestroyShaderModule(_device, colorMeshFragShader, nullptr);
    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        // TODO auto unload material
        vkDestroyPipeline(_device, meshPipeline, nullptr);
        vkDestroyPipelineLayout(_device, meshPipelineLayout, nullptr);
    });
}

void VulkanEngine::init_scene() {
    //create a sampler for the texture
    VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    VkSampler blockySampler;
    vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);

    _mainDeletionQueue.push_function([=] {
        vkDestroySampler(_device, blockySampler, nullptr);
    });

    Material *texturedMat = get_material("default_mesh");

    //allocate the descriptor set for single-texture to use on the material
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pNext = nullptr;
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_singleTextureSetLayout;

    vkAllocateDescriptorSets(_device, &allocInfo, &texturedMat->textureSet);

    //write to the descriptor set so that it points to our empire_diffuse texture
    VkDescriptorImageInfo imageBufferInfo;
    imageBufferInfo.sampler = blockySampler;
    imageBufferInfo.imageView = _loadedTextures["empire_diffuse"].imageView;
    imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet texture1 = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                                   texturedMat->textureSet, &imageBufferInfo, 0);

    vkUpdateDescriptorSets(_device, 1, &texture1, 0, nullptr);


    RenderObject monkey;
    monkey.mesh = get_mesh("monkey");
    monkey.material = texturedMat;
    monkey.transformMatrix = glm::mat4{1.0f};

    _renderables.push_back(monkey);

    RenderObject empire{
            get_mesh("empire"),
            texturedMat,
            glm::mat4{1.0f}
    };

    _renderables.push_back(empire);

}

void VulkanEngine::init_descriptors() {
    std::vector<VkDescriptorPoolSize> sizes{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         10},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         10},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.maxSets = 10;
    poolInfo.poolSizeCount = (uint32_t) sizes.size();
    poolInfo.pPoolSizes = sizes.data();

    vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool);

    auto cameraBinding = vkinit::descriptor_set_layout_binding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

    const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));
    _sceneParameterBuffer = create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto sceneBinding = vkinit::descriptor_set_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                              1);

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {cameraBinding, sceneBinding};

    VkDescriptorSetLayoutCreateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setInfo.flags = 0;
    setInfo.pNext = nullptr;
    setInfo.bindingCount = bindings.size();
    setInfo.pBindings = bindings.data();

    vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);

    auto objectBinding = vkinit::descriptor_set_layout_binding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT,
            0);

    std::array<VkDescriptorSetLayoutBinding, 1> set2Bindings = {objectBinding};

    VkDescriptorSetLayoutCreateInfo set2Info = {};
    set2Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set2Info.flags = 0;
    set2Info.pNext = nullptr;
    set2Info.bindingCount = 1;
    set2Info.pBindings = &objectBinding;

    vkCreateDescriptorSetLayout(_device, &set2Info, nullptr, &_objectSetLayout);

    VkDescriptorSetLayoutBinding textureBind = vkinit::descriptor_set_layout_binding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

    VkDescriptorSetLayoutCreateInfo set3info = {};
    set3info.bindingCount = 1;
    set3info.flags = 0;
    set3info.pNext = nullptr;
    set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set3info.pBindings = &textureBind;

    vkCreateDescriptorSetLayout(_device, &set3info, nullptr, &_singleTextureSetLayout);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _objectSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _singleTextureSetLayout, nullptr);
    });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        auto &frameData = _frames[i];

        const int MAX_OBJECTS = 10000;
        frameData.objectBuffer = create_buffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               VMA_MEMORY_USAGE_CPU_TO_GPU);

        auto &cameraBuffer = _frames[i].cameraBuffer;
        cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                     VMA_MEMORY_USAGE_CPU_TO_GPU);
        //allocate one descriptor set for each frameData
        VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptor_alloc_info(_descriptorPool, 1, &_globalSetLayout);
        vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].globalDescriptor);

        VkDescriptorSetAllocateInfo objectAllocInfo = vkinit::descriptor_alloc_info(_descriptorPool, 1,
                                                                                    &_objectSetLayout);
        vkAllocateDescriptorSets(_device, &objectAllocInfo, &_frames[i].objectDescriptor);

        VkDescriptorBufferInfo cameraInfo;
        cameraInfo.buffer = frameData.cameraBuffer._buffer;
        cameraInfo.offset = 0;
        cameraInfo.range = sizeof(GPUCameraData);

        VkDescriptorBufferInfo sceneInfo;
        sceneInfo.buffer = _sceneParameterBuffer._buffer;
        sceneInfo.offset = 0;
        sceneInfo.range = sizeof(GPUSceneData);

        VkDescriptorBufferInfo objectBufferInfo;
        objectBufferInfo.buffer = frameData.objectBuffer._buffer;
        objectBufferInfo.offset = 0;
        objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

        VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                frameData.globalDescriptor,
                &cameraInfo,
                0
        );

        VkWriteDescriptorSet sceneWrite = vkinit::write_descriptor_buffer(
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                frameData.globalDescriptor,
                &sceneInfo,
                1
        );

        VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                frameData.objectDescriptor,
                &objectBufferInfo,
                0);

        std::array<VkWriteDescriptorSet, 3> setWrites = {cameraWrite, sceneWrite, objectWrite};

        vkUpdateDescriptorSets(_device, setWrites.size(), setWrites.data(), 0, nullptr);

        _mainDeletionQueue.push_function([&]() {
            vmaDestroyBuffer(_allocator, cameraBuffer._buffer, cameraBuffer._allocation);
            vmaDestroyBuffer(_allocator, frameData.objectBuffer._buffer, frameData.objectBuffer._allocation);
        });
    }

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);
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
    //we now use all of the info structs we have been writing into this one to create the pipeline
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
    pipelineInfo.pDepthStencilState = &_depthStencil;

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

    Mesh triangleMesh{};
    Mesh monkeyMesh{};
    Mesh lostEmpire{};
    //make the array 3 vertices long
    triangleMesh._vertices.resize(3);

    //vertex positions
    triangleMesh._vertices[0].position = {1.f, 1.f, 0.0f};
    triangleMesh._vertices[1].position = {-1.f, 0.5f, 0.0f};
    triangleMesh._vertices[2].position = {0.f, -1.f, 0.0f};

    //vertex colors, all green
    triangleMesh._vertices[0].color = {0.f, 1.f, 0.0f}; //pure green
    triangleMesh._vertices[1].color = {0.f, 1.f, 0.0f}; //pure green
    triangleMesh._vertices[2].color = {0.f, 1.f, 0.0f}; //pure green

    triangleMesh._indices = {0, 1, 2};

    monkeyMesh.load_from_obj("../assets/monkey_smooth.obj");
    lostEmpire.load_from_obj("../assets/lost_empire.obj");

    upload_mesh(triangleMesh);
    upload_mesh(monkeyMesh);
    upload_mesh(lostEmpire);

    _meshes["monkey"] = monkeyMesh;
    _meshes["triangle"] = triangleMesh;
    _meshes["empire"] = lostEmpire;
}

void VulkanEngine::load_images() {
    Texture lostEmpire;
    vkutil::load_image_from_file(*this, "../assets/lost_empire-RGBA.png", lostEmpire.image);

    auto imageInfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image._image,
                                                   VK_IMAGE_ASPECT_COLOR_BIT);
    vkCreateImageView(_device, &imageInfo, nullptr, &lostEmpire.imageView);

    _loadedTextures["empire_diffuse"] = lostEmpire;

    _mainDeletionQueue.push_function([=, this] {
        vkDestroyImageView(_device, lostEmpire.imageView, nullptr);
    });
}

void VulkanEngine::copy_buffer_cpu2gpu(size_t bufferSize, void *pData, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                       VkDeviceSize srcOffset) {
    AllocatedBuffer stagingBuffer = create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  VMA_MEMORY_USAGE_CPU_ONLY);
    write_buffer(pData, bufferSize, stagingBuffer._allocation);

    immediate_submit([=](VkCommandBuffer cmd) {
        VkBufferCopy copy;
        copy.dstOffset = dstOffset;
        copy.srcOffset = srcOffset;
        copy.size = bufferSize;
        vkCmdCopyBuffer(cmd, stagingBuffer._buffer, dstBuffer, 1, &copy);
    });
    vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);
}

void VulkanEngine::upload_mesh(Mesh &mesh) {
    const size_t vertexBufferSize = mesh._vertices.size() * sizeof(Vertex);
    mesh._vertexBuffer = create_buffer(vertexBufferSize,
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY);
    copy_buffer_cpu2gpu(vertexBufferSize, mesh._vertices.data(), mesh._vertexBuffer._buffer);

    _mainDeletionQueue.push_function([=, this]() {
        vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
    });


    auto indexSize = mesh._indices.size() * sizeof(uint32_t);
    mesh._indexBuffer = create_buffer(indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_GPU_ONLY);
    copy_buffer_cpu2gpu(indexSize, mesh._indices.data(), mesh._indexBuffer._buffer);
    _mainDeletionQueue.push_function([=, this]() {
        vmaDestroyBuffer(_allocator, mesh._indexBuffer._buffer, mesh._indexBuffer._allocation);
    });
}

void VulkanEngine::write_buffer(void *pData, VkDeviceSize size, VmaAllocation allocation, size_t offset) {
    char *data;
    // 将映射到 GPU 内存映射到 CPU 地址空间
    vmaMapMemory(_allocator, allocation, (void **) &data);
    // 将数据拷贝到映射了的 GPU 内存里
    memcpy(data + offset, pData, size);
    // 取消映射
    vmaUnmapMemory(_allocator, allocation);
}

AllocatedBuffer VulkanEngine::create_buffer(size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) const {
    VkBufferCreateInfo bufferInfo = vkinit::buffer_create_info(size, usage);
    VmaAllocationCreateInfo vmaAllocInfo{};
    vmaAllocInfo.usage = memoryUsage;

    AllocatedBuffer newBuffer;
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
                             &newBuffer._buffer,
                             &newBuffer._allocation,
                             nullptr));
    return newBuffer;
}


Material *VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name) {
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = layout;
    _materials[name] = mat;
    return &_materials[name];
}

Material *VulkanEngine::get_material(const std::string &name) {
    auto it = _materials.find(name);
    if (it == _materials.end()) return nullptr;
    else return &(*it).second;
}

Mesh *VulkanEngine::get_mesh(const std::string &name) {
    auto it = _meshes.find(name);
    if (it == _meshes.end()) return nullptr;
    else return &(*it).second;
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject *first, int count) {
//    glm::vec3 camPos = {0.f, -6.f, 20.f};
//    glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
//    glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
//    projection[1][1] *= -1;
    auto cameraData = _camera->gpu_data();

    write_buffer(&cameraData, sizeof(GPUCameraData), get_current_frame().cameraBuffer._allocation);

    auto frameIndex = _frameNumber % FRAME_OVERLAP;
    float framed = _frameNumber / 120.f;
    _sceneParameters.ambientColor = {(sin(framed) + 1.f) / 2.0f, 0, (cos(framed) + 1) / 2.0f, 1};
    write_buffer(&_sceneParameters, sizeof(GPUSceneData), _sceneParameterBuffer._allocation, pad_uniform_buffer_size(
            sizeof(GPUSceneData)) * frameIndex);

    // Write Object Data Buffer
    void *objectData;
    vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);
    auto *objectSSBO = (GPUObjectData *) objectData;
    for (int i = 0; i < count; ++i) {
        RenderObject &object = first[i];
        objectSSBO[i].modelMatrix = object.transformMatrix;
    }
    vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);

    Mesh *lastMesh = nullptr;
    Material *lastMaterial = nullptr;
    for (int i = 0; i < count; ++i) {
        RenderObject &object = first[i];
        // If the material changed, we change the pipeline to the new material's
        if (object.material != lastMaterial) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
            lastMaterial = object.material;

            uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
            vkCmdBindDescriptorSets(cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    object.material->pipelineLayout,
                                    0,
                                    1,
                                    &get_current_frame().globalDescriptor,
                                    1,
                                    &uniform_offset);

            // Bind Object Descriptor Set
            vkCmdBindDescriptorSets(cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    object.material->pipelineLayout,
                                    1,
                                    1,
                                    &get_current_frame().objectDescriptor,
                                    0,
                                    nullptr);

            if (object.material->textureSet != VK_NULL_HANDLE) {
                //texture descriptor
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 2, 1,
                                        &object.material->textureSet, 0, nullptr);

            }
        }

        MeshPushConstants constants;
        constants.render_matrix = object.transformMatrix;

        vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(MeshPushConstants), &constants);

        if (object.mesh != lastMesh) {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
            vkCmdBindIndexBuffer(cmd, object.mesh->_indexBuffer._buffer, 0, VK_INDEX_TYPE_UINT32);
            lastMesh = object.mesh;
        }

        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(object.mesh->_indices.size()), 1, 0, 0, i);
    }
}

FrameData &VulkanEngine::get_current_frame() {
    return _frames[_frameNumber % FRAME_OVERLAP];
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize) {
    size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
    size_t alignedSize = originalSize;
    if (minUboAlignment > 0) {
        alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
    }
    return alignedSize;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer)> &&function) {
    auto cmd = _uploadContext.commandBuffer;
    auto cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo = vkinit::submit_info(&cmd);
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submitInfo, _uploadContext.uploadFence));

    vkWaitForFences(_device, 1, &_uploadContext.uploadFence, true, 9999999999);
    vkResetFences(_device, 1, &_uploadContext.uploadFence);

    vkResetCommandPool(_device, _uploadContext.commandPool, 0);
}


//region Camera
GPUCameraData Camera::gpu_data() {

    glm::mat4 view = glm::lookAtRH(position, position + glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 1.f, 0.f));;

    glm::mat4 projection = glm::perspective(fovy, aspect, zNear, zFar);
    projection[1][1] *= -1;
    return GPUCameraData{
            view,
            projection,
            projection * view,
    };
}

//endregion
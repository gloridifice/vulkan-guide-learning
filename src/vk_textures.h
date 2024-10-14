//
// Created by Koiro on 13/10/2024.
//
#pragma once

#ifndef VULKAN_GUIDE_VK_TEXTURES_H
#define VULKAN_GUIDE_VK_TEXTURES_H

#include "vk_types.h"
#include "vk_engine.h"

namespace vkutil{
    bool load_image_from_file(VulkanEngine& engine, const char* file, AllocatedImage& outImage);

}

#endif //VULKAN_GUIDE_VK_TEXTURES_H

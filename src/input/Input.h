//
// Created by Koiro on 14/10/2024.
//
#pragma once

#ifndef VULKAN_GUIDE_INPUT_H
#define VULKAN_GUIDE_INPUT_H

#include "array"
#include "SDL.h"
#include "unordered_map"
class Input {
public:
    static Input* INSTANCE;
    std::unordered_map<SDL_Keycode , bool> _keys;

    Input();
};


#endif //VULKAN_GUIDE_INPUT_H

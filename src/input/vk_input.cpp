//
// Created by Koiro on 14/10/2024.
//

#include "vk_input.h"
#include "iostream"

bool vkinput::get_key(SDL_Keycode keyCode) {
    return Input::INSTANCE->_keys[keyCode];
}

#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/vec3.hpp>
#include "glm/vec2.hpp"
#include <glm/gtx/hash.hpp>

struct VertexInputDescription {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;

    static VertexInputDescription get_vertex_description();

    bool operator==(const Vertex& other) const {
        return position == other.position && color == other.color && color == other.color;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^
                     (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec3>()(vertex.normal) << 1);
        }
    };
}


struct Mesh {
    std::vector<Vertex> _vertices;
    std::vector<uint32_t> _indices;
    AllocatedBuffer _vertexBuffer;
    AllocatedBuffer _indexBuffer;

    bool load_from_obj(const char* filename);
};


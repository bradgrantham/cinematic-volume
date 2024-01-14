#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <array>
#include <filesystem>
#include <thread>
// #include <execution>

#include <cstring>
#include <cassert>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "vectormath.h"
#include "manipulator.h"

#if defined(_WIN32)
#define PLATFORM_WINDOWS
#elif defined(__linux__)
#define PLATFORM_LINUX
#elif defined(__APPLE__) && defined(__MACH__)
#define PLATFORM_MACOS
#else
#error Platform not supported.
#endif

static constexpr uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

template <typename T>
size_t ByteCount(const std::vector<T>& v) { return sizeof(T) * v.size(); }

#define STR(f) #f

std::map<VkResult, std::string> MapVkResultToName =
{
    {VK_ERROR_OUT_OF_HOST_MEMORY, "OUT_OF_HOST_MEMORY"},
    {VK_ERROR_OUT_OF_DEVICE_MEMORY, "OUT_OF_DEVICE_MEMORY"},
    {VK_ERROR_INITIALIZATION_FAILED, "INITIALIZATION_FAILED"},
    {VK_ERROR_DEVICE_LOST, "DEVICE_LOST"},
    {VK_ERROR_MEMORY_MAP_FAILED, "MEMORY_MAP_FAILED"},
    {VK_ERROR_LAYER_NOT_PRESENT, "LAYER_NOT_PRESENT"},
    {VK_ERROR_EXTENSION_NOT_PRESENT, "EXTENSION_NOT_PRESENT"},
    {VK_ERROR_FEATURE_NOT_PRESENT, "FEATURE_NOT_PRESENT"},
};

#define VK_CHECK(f) \
{ \
    VkResult result = (f); \
    static const std::set<VkResult> okay{VK_SUCCESS, VK_SUBOPTIMAL_KHR, VK_THREAD_IDLE_KHR, VK_THREAD_DONE_KHR, VK_OPERATION_DEFERRED_KHR, VK_OPERATION_NOT_DEFERRED_KHR}; \
    if(!okay.contains(result)) { \
	if(MapVkResultToName.count(f) > 0) { \
	    std::cerr << "VkResult from " STR(f) " was " << MapVkResultToName[result] << " at line " << __LINE__ << "\n"; \
	} else { \
	    std::cerr << "VkResult from " STR(f) " was " << result << " at line " << __LINE__ << "\n"; \
        } \
	exit(EXIT_FAILURE); \
    } \
}

// Adapted from vkcube.cpp
VkSurfaceFormatKHR PickSurfaceFormat(std::vector<VkSurfaceFormatKHR>& surfaceFormats)
{
    // Prefer non-SRGB formats...
    int which = 0;
    for (const auto& surfaceFormat: surfaceFormats) {
        const VkFormat format = surfaceFormat.format;

        if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_B8G8R8A8_UNORM) {
            return surfaceFormats[which];
        }
        which++;
    }

    printf("Can't find our preferred formats... Falling back to first exposed format. Rendering may be incorrect.\n");

    return surfaceFormats[0];
}

VkCommandPool GetCommandPool(VkDevice device, uint32_t queue)
{
    VkCommandPoolCreateInfo create_command_pool{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue,
    };
    VkCommandPool command_pool;
    VK_CHECK(vkCreateCommandPool(device, &create_command_pool, nullptr, &command_pool));
    return command_pool;
}

VkCommandBuffer GetCommandBuffer(VkDevice device, VkCommandPool command_pool)
{
    VkCommandBufferAllocateInfo cmdBufAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command_buffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &command_buffer));
    return command_buffer;
}

void BeginCommandBuffer(VkCommandBuffer command_buffer)
{
    VkCommandBufferBeginInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };
    VK_CHECK(vkBeginCommandBuffer(command_buffer, &info));
}

void FlushCommandBuffer(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer)
{
    VkSubmitInfo submitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = 0,
    };
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

    // Submit to the queue
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

    vkDestroyFence(device, fence, nullptr);
}

uint32_t GetMemoryTypeIndex(VkPhysicalDeviceMemoryProperties memory_properties, uint32_t type_bits, VkMemoryPropertyFlags properties)
{
    // Adapted from Sascha Willem's 
    // Iterate over all memory types available for the device used in this example
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
	if (type_bits & (1 << i)) {
	    if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
		return i;
            }
        }
    }

    throw std::runtime_error("Could not find a suitable memory type!");
}

static constexpr uint32_t NO_QUEUE_FAMILY = 0xffffffff;

struct Vertex
{
    vec3 v;
    vec3 n;
    vec4 c;
    vec3 t;

    Vertex(const vec3& v, const vec3& n, const vec4& c, const vec2& t) :
        v(v),
        n(n),
        c(c),
        t(t)
    { }
    Vertex() {}

    static std::vector<VkVertexInputAttributeDescription> GetVertexInputAttributeDescription()
    {
        std::vector<VkVertexInputAttributeDescription> vertex_input_attributes;
        vertex_input_attributes.push_back({0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, v)});
        vertex_input_attributes.push_back({1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, n)});
        vertex_input_attributes.push_back({2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, c)});
        vertex_input_attributes.push_back({3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, t)});
        return vertex_input_attributes;
    }
};

struct Buffer
{
    VkDevice device;
    VkDeviceMemory mem { VK_NULL_HANDLE };
    VkBuffer buf { VK_NULL_HANDLE };
    void* mapped { nullptr };

    void Create(VkPhysicalDevice physical_device, VkDevice device_, size_t size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags properties)
    {
        Release();

        device = device_;

        VkBufferCreateInfo create_buffer {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage_flags,
        };

        VK_CHECK(vkCreateBuffer(device, &create_buffer, nullptr, &buf));

        VkMemoryRequirements memory_req;
        vkGetBufferMemoryRequirements(device, buf, &memory_req);

        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

        uint32_t memoryTypeIndex = GetMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, properties);
        VkMemoryAllocateInfo memory_alloc {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memory_req.size,
            .memoryTypeIndex = memoryTypeIndex,
        };
        VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
    }

    void Release()
    {
        if(mapped) {
            vkUnmapMemory(device, mem);
            mapped = nullptr;
        }
        if(mem != VK_NULL_HANDLE) {
            vkFreeMemory(device, mem, nullptr);
            mem = VK_NULL_HANDLE;
            vkDestroyBuffer(device, buf, nullptr);
            buf = VK_NULL_HANDLE;
        }
    }
};

struct UniformBuffer
{
    uint32_t binding;
    VkShaderStageFlags stageFlags;
    size_t size;

    UniformBuffer(int binding, VkShaderStageFlags stageFlags, size_t size) :
        binding(binding),
        stageFlags(stageFlags),
        size(size)
    {}
};

struct ImageSampler
{
    uint32_t binding;
    VkShaderStageFlags stageFlags;
    ImageSampler(int binding, VkShaderStageFlags stageFlags) :
        binding(binding),
        stageFlags(stageFlags)
    {}
};

struct ShadingUniforms
{
    vec3 specular_color;
    float shininess;
};

struct VertexUniforms
{
    mat4f modelview;
    mat4f modelview_normal;
    mat4f projection;
};

struct FragmentUniforms
{
    vec3 light_position;
    float pad; // XXX grr - why can't I get this right with "pragma align"?
    vec3 light_color;
    float pad2;
};

VkInstance CreateInstance(bool enable_validation)
{
    VkInstance instance;
    std::set<std::string> extension_set;
    std::set<std::string> layer_set;

    uint32_t glfw_reqd_extension_count;
    const char** glfw_reqd_extensions = glfwGetRequiredInstanceExtensions(&glfw_reqd_extension_count);
    extension_set.insert(glfw_reqd_extensions, glfw_reqd_extensions + glfw_reqd_extension_count);

    extension_set.insert(VK_KHR_SURFACE_EXTENSION_NAME);

#if defined(PLATFORM_WINDOWS)
    extension_set.insert("VK_KHR_win32_surface");
#elif defined(PLATFORM_LINUX)
    extension_set.insert(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(PLATFORM_MACOS)
    // extension_set.insert(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
    extension_set.insert("VK_MVK_macos_surface");
    extension_set.insert(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    if(enable_validation) {
	layer_set.insert("VK_LAYER_KHRONOS_validation");
    }

    // Make this an immediately invoked lambda so I know the c_str() I called remains
    // valid through the scope of this lambda.
    [&](const std::set<std::string> &extension_set, const std::set<std::string> &layer_set) {

        std::vector<const char*> extensions;
        std::vector<const char*> layers;

	for(auto& s: extension_set) {
	    extensions.push_back(s.c_str());
        }

	for(auto& s: layer_set) {
	    layers.push_back(s.c_str());
        }

	VkApplicationInfo app_info {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "vulkan-test-grantham",
            .pEngineName = "vulkan-test-grantham",
            .apiVersion = VK_API_VERSION_1_2,
        };

	VkInstanceCreateInfo create {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#if defined(PLATFORM_MACOS)
            .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
#endif
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

	VK_CHECK(vkCreateInstance(&create, nullptr, &instance));

    }(extension_set, layer_set);

    return instance;
}

VkPhysicalDevice ChoosePhysicalDevice(VkInstance instance)
{
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
    std::vector<VkPhysicalDevice> physical_devices(gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices.data()));

    return physical_devices[0];
}

const std::vector<std::string> DeviceTypeDescriptions = {
    "other",
    "integrated GPU",
    "discrete GPU",
    "virtual GPU",
    "CPU",
    "unknown",
};

std::map<uint32_t, std::string> MemoryPropertyBitToNameMap = {
    {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
    {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE"},
    {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"},
    {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED"},
    {VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZILY_ALLOCATED"},
};

void PrintMemoryPropertyBits(VkMemoryPropertyFlags flags)
{
    bool add_or = false;
    for(auto& bit : MemoryPropertyBitToNameMap) {
	if(flags & bit.first) {
	    printf("%s%s", add_or ? " | " : "", bit.second.c_str());
	    add_or = true;
	}
    }
}

void PrintDeviceInformation(VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties &memory_properties)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    printf("Physical Device Information\n");
    printf("    API     %d.%d.%d\n", properties.apiVersion >> 22, (properties.apiVersion >> 12) & 0x3ff, properties.apiVersion & 0xfff);
    printf("    driver  %X\n", properties.driverVersion);
    printf("    vendor  %X\n", properties.vendorID);
    printf("    device  %X\n", properties.deviceID);
    printf("    name    %s\n", properties.deviceName);
    printf("    type    %s\n", DeviceTypeDescriptions[std::min(5, (int)properties.deviceType)].c_str());

    uint32_t ext_count;

    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, exts.data());
    printf("    extensions:\n");
    for(const auto& ext: exts) {
	printf("        %s\n", ext.extensionName);
    }

    // VkPhysicalDeviceLimits              limits;
    // VkPhysicalDeviceSparseProperties    sparseProperties;
    //
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    int queue_index = 0;
    for(const auto& queue_family: queue_families) {
        printf("queue %d:\n", queue_index++);
        printf("    flags:                       %04X\n", queue_family.queueFlags);
        printf("    queueCount:                  %d\n", queue_family.queueCount);
        printf("    timestampValidBits:          %d\n", queue_family.timestampValidBits);
        printf("    minImageTransferGranularity: (%d, %d, %d)\n",
            queue_family.minImageTransferGranularity.width,
            queue_family.minImageTransferGranularity.height,
            queue_family.minImageTransferGranularity.depth);
    }

    for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        printf("memory type %d: flags ", i);
        PrintMemoryPropertyBits(memory_properties.memoryTypes[i].propertyFlags);
        printf("\n");
    }
}

std::vector<uint8_t> GetFileContents(FILE *fp)
{
    long int start = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long int end = ftell(fp);
    fseek(fp, start, SEEK_SET);

    std::vector<uint8_t> data(end - start);
    [[maybe_unused]] size_t result = fread(data.data(), 1, end - start, fp);
    if(result != static_cast<size_t>(end - start)) {
        fprintf(stderr, "short read\n");
        abort();
    }

    return data;
}

std::vector<uint32_t> GetFileAsCode(const std::string& filename) 
{
    FILE *fp = fopen(filename.c_str(), "rb");
    if(fp == nullptr) {
        fprintf(stderr, "couldn't open \"%s\" for reading\n", filename.c_str());
        abort();
    }
    std::vector<uint8_t> text = GetFileContents(fp);
    fclose(fp);

    std::vector<uint32_t> code((text.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    memcpy(code.data(), text.data(), text.size()); // XXX this is probably UB that just happens to work... also maybe endian

    return code;
}

VkShaderModule CreateShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
    VkShaderModuleCreateInfo shader_create {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .flags = 0,
        .codeSize = code.size() * sizeof(code[0]),
        .pCode = code.data(),
    };

    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &shader_create, NULL, &module));
    return module;
}


VkDevice CreateDevice(VkPhysicalDevice physical_device, const std::vector<const char*>& extensions, uint32_t queue_family)
{
    float queue_priorities = 1.0f;

    VkDeviceQueueCreateInfo create_queues {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .flags = 0,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priorities,
    };

    VkDeviceCreateInfo create {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &create_queues,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    VkDevice device;
    VK_CHECK(vkCreateDevice(physical_device, &create, nullptr, &device));
    return device;
}

VkSampler CreateSampler(VkDevice device, VkSamplerMipmapMode mipMode, VkSamplerAddressMode wrapMode)
{
    VkSamplerCreateInfo create_sampler {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = mipMode,
        .addressModeU = wrapMode,
        .addressModeV = wrapMode,
        .addressModeW = wrapMode,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 0.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
        // .borderColor
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler textureSampler;
    VK_CHECK(vkCreateSampler(device, &create_sampler, nullptr, &textureSampler));
    return textureSampler;
}

VkImageView CreateImageView(VkDevice device, VkFormat format, VkImage image, VkImageAspectFlags aspect)
{
    VkImageViewCreateInfo imageViewCreate {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .flags = 0,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components {VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    VkImageView imageView;
    VK_CHECK(vkCreateImageView(device, &imageViewCreate, nullptr, &imageView));
    return imageView;
}

VkFramebuffer CreateFramebuffer(VkDevice device, const std::vector<VkImageView>& imageviews, VkRenderPass render_pass, uint32_t width, uint32_t height)
{
    VkFramebufferCreateInfo framebufferCreate {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .flags = 0,
        .renderPass = render_pass,
        .attachmentCount = static_cast<uint32_t>(imageviews.size()),
        .pAttachments = imageviews.data(),
        .width = width,
        .height = height,
        .layers = 1,
    };
    VkFramebuffer framebuffer;
    VK_CHECK(vkCreateFramebuffer(device, &framebufferCreate, nullptr, &framebuffer));
    return framebuffer;
}

void PrintImplementationInformation()
{
    uint32_t ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());
    printf("Vulkan instance extensions:\n");
    for (const auto& ext: exts) {
        printf("\t%s, %08X\n", ext.extensionName, ext.specVersion);
    }
}

uint32_t FindQueueFamily(VkPhysicalDevice physical_device, VkQueueFlags queue_flags)
{
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    for(uint32_t i = 0; i < queue_family_count; i++) {
        if((queue_families[i].queueFlags & queue_flags) == queue_flags) {
            return i;
        }
    }
    return NO_QUEUE_FAMILY;
}

// XXX This need to be refactored
void CreateGeometryBuffers(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Buffer* vertex_buffer, Buffer* index_buffer)
{
    uint32_t transfer_queue = FindQueueFamily(physical_device, VK_QUEUE_TRANSFER_BIT);
    if(transfer_queue == NO_QUEUE_FAMILY) {
        throw std::runtime_error("couldn't find a transfer queue\n");
    }

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    // host-writable memory and buffers
    Buffer vertex_staging;
    Buffer index_staging;

    VkBufferCreateInfo create_staging_buffer{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };

    create_staging_buffer.size = ByteCount(vertices);
    VK_CHECK(vkCreateBuffer(device, &create_staging_buffer, nullptr, &vertex_staging.buf));

    VkMemoryRequirements memory_req{};
    vkGetBufferMemoryRequirements(device, vertex_staging.buf, &memory_req);

    uint32_t memoryTypeIndex = GetMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo memory_alloc {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_req.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &vertex_staging.mem));

    VK_CHECK(vkMapMemory(device, vertex_staging.mem, 0, memory_alloc.allocationSize, 0, &vertex_staging.mapped));
    memcpy(vertex_staging.mapped, vertices.data(), ByteCount(vertices));
    vkUnmapMemory(device, vertex_staging.mem);

    VK_CHECK(vkBindBufferMemory(device, vertex_staging.buf, vertex_staging.mem, 0));

    create_staging_buffer.size = ByteCount(indices);
    VK_CHECK(vkCreateBuffer(device, &create_staging_buffer, nullptr, &index_staging.buf));

    vkGetBufferMemoryRequirements(device, index_staging.buf, &memory_req);

    memory_alloc.allocationSize = memory_req.size;
    // Find the type which this memory requires which is visible to the
    // CPU and also coherent, so when we unmap it it will be immediately
    // visible to the GPU
    memory_alloc.memoryTypeIndex = GetMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &index_staging.mem));

    VK_CHECK(vkMapMemory(device, index_staging.mem, 0, memory_alloc.allocationSize, 0, &index_staging.mapped));
    memcpy(index_staging.mapped, indices.data(), ByteCount(indices));
    vkUnmapMemory(device, index_staging.mem);

    VK_CHECK(vkBindBufferMemory(device, index_staging.buf, index_staging.mem, 0));

    VkBufferCreateInfo create_vertex_buffer {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };

    create_vertex_buffer.size = ByteCount(vertices);
    VK_CHECK(vkCreateBuffer(device, &create_vertex_buffer, nullptr, &vertex_buffer->buf));

    vkGetBufferMemoryRequirements(device, vertex_buffer->buf, &memory_req);

    memory_alloc.allocationSize = memory_req.size;
    memory_alloc.memoryTypeIndex = GetMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &vertex_buffer->mem));
    VK_CHECK(vkBindBufferMemory(device, vertex_buffer->buf, vertex_buffer->mem, 0));

    VkBufferCreateInfo create_index_buffer {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };

    create_index_buffer.size = ByteCount(indices);
    VK_CHECK(vkCreateBuffer(device, &create_index_buffer, nullptr, &index_buffer->buf));

    vkGetBufferMemoryRequirements(device, index_buffer->buf, &memory_req);

    memory_alloc.allocationSize = memory_req.size;
    memory_alloc.memoryTypeIndex = GetMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &index_buffer->mem));
    VK_CHECK(vkBindBufferMemory(device, index_buffer->buf, index_buffer->mem, 0));

    VkCommandPool command_pool = GetCommandPool(device, transfer_queue);
    VkCommandBuffer transfer_commands = GetCommandBuffer(device, command_pool);

    BeginCommandBuffer(transfer_commands);
    {
        VkBufferCopy copy {0, 0, ByteCount(vertices)};
        vkCmdCopyBuffer(transfer_commands, vertex_staging.buf, vertex_buffer->buf, 1, &copy);
    }
    {
        VkBufferCopy copy {0, 0, ByteCount(indices)};
        vkCmdCopyBuffer(transfer_commands, index_staging.buf, index_buffer->buf, 1, &copy);
    }
    VK_CHECK(vkEndCommandBuffer(transfer_commands));

    FlushCommandBuffer(device, queue, transfer_commands);

    vkFreeCommandBuffers(device, command_pool, 1, &transfer_commands);
    vkDestroyBuffer(device, vertex_staging.buf, nullptr);
    vkDestroyBuffer(device, index_staging.buf, nullptr);
    vkFreeMemory(device, vertex_staging.mem, nullptr);
    vkFreeMemory(device, index_staging.mem, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
}

template <class TEXTURE>
void CreateDeviceTextureImage(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, TEXTURE texture, VkImage* textureImage, VkDeviceMemory* textureMemory, VkImageUsageFlags usage_flags, VkImageLayout final_layout)
{
    Buffer staging_buffer;
    staging_buffer.Create(physical_device, device, texture->GetSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkMapMemory(device, staging_buffer.mem, 0, texture->GetSize(), 0, &staging_buffer.mapped));
    memcpy(staging_buffer.mapped, texture->GetData(), texture->GetSize());
    vkUnmapMemory(device, staging_buffer.mem);

    auto [image, memory] = CreateBound2DImage(physical_device, device, texture->GetVulkanFormat(), static_cast<uint32_t>(texture->GetWidth()), static_cast<uint32_t>(texture->GetHeight()), usage_flags, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    *textureImage = image;
    *textureMemory = memory;

    uint32_t transfer_queue = FindQueueFamily(physical_device, VK_QUEUE_TRANSFER_BIT);
    if(transfer_queue == NO_QUEUE_FAMILY) {
        fprintf(stderr, "couldn't find a transfer queue\n");
        abort();
    }
    VkCommandPool command_pool = GetCommandPool(device, transfer_queue);
    VkCommandBuffer transfer_commands = GetCommandBuffer(device, command_pool);

    BeginCommandBuffer(transfer_commands);

    VkImageMemoryBarrier transfer_dst_optimal {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image = *textureImage,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(transfer_commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transfer_dst_optimal);

    VkBufferImageCopy copy {
        .bufferOffset = 0,
        .bufferRowLength = static_cast<uint32_t>(texture->GetWidth()),
        .bufferImageHeight = static_cast<uint32_t>(texture->GetHeight()),
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {static_cast<uint32_t>(texture->GetWidth()), static_cast<uint32_t>(texture->GetHeight()), 1},
    };
    vkCmdCopyBufferToImage(transfer_commands, staging_buffer.buf, *textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier shader_read_optimal {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = final_layout,
        .image = *textureImage,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(transfer_commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &shader_read_optimal);

    VK_CHECK(vkEndCommandBuffer(transfer_commands));

    FlushCommandBuffer(device, queue, transfer_commands);
    vkFreeCommandBuffers(device, command_pool, 1, &transfer_commands);
    vkDestroyBuffer(device, staging_buffer.buf, nullptr);
    vkFreeMemory(device, staging_buffer.mem, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
}

struct RGBA8UNormImage
{
private:
    int width;
    int height;
    std::vector<uint8_t> rgba8_unorm;

public:
    RGBA8UNormImage(int width, int height, std::vector<uint8_t>& rgba8_unorm) :
        width(width),
        height(height),
        rgba8_unorm(std::move(rgba8_unorm))
    {}

    VkFormat GetVulkanFormat() { return VK_FORMAT_R8G8B8A8_UNORM; }
    int GetWidth() { return width; }
    int GetHeight() { return height; }
    void* GetData() { return rgba8_unorm.data(); }
    size_t GetSize() { return rgba8_unorm.size(); }
};

struct Drawable
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    int triangleCount;
    aabox bounds;

    float specular_color[4];
    float shininess;

    std::shared_ptr<RGBA8UNormImage> texture;

    VkImage textureImage { VK_NULL_HANDLE };
    VkDeviceMemory textureMemory { VK_NULL_HANDLE };
    VkImageView textureImageView { VK_NULL_HANDLE };
    VkSampler textureSampler { VK_NULL_HANDLE };

    constexpr static int VERTEX_BUFFER = 0;
    constexpr static int INDEX_BUFFER = 1;
    typedef std::array<Buffer, 2> DrawableBuffersOnDevice;
    std::map<VkDevice, DrawableBuffersOnDevice> buffers_by_device;

    Drawable(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
        float specular_color[4], float shininess, std::shared_ptr<RGBA8UNormImage> texture) :
            vertices(vertices),
            indices(indices),
            shininess(shininess),
            texture(texture)
    {
        this->specular_color[0] = specular_color[0];
        this->specular_color[1] = specular_color[1];
        this->specular_color[2] = specular_color[2];
        this->specular_color[3] = specular_color[3];
        triangleCount = static_cast<int>(indices.size() / 3);
        for(uint32_t i = 0; i < vertices.size(); i++) {
            bounds += vertices[i].v;
        }
    }

    void CreateDeviceData(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue)
    {
        if(texture) {
            CreateDeviceTextureImage(physical_device, device, queue, texture, &textureImage, &textureMemory, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            textureSampler = CreateSampler(device, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
            textureImageView = CreateImageView(device, texture->GetVulkanFormat(), textureImage, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        Buffer vertex_buffer, index_buffer;
        CreateGeometryBuffers(physical_device, device, queue, vertices, indices, &vertex_buffer, &index_buffer);
        buffers_by_device.insert({device, {vertex_buffer, index_buffer}});
    }

    void BindForDraw(VkDevice device, VkCommandBuffer cmdbuf)
    {
        VkDeviceSize offset = 0;
        auto buffers = buffers_by_device.at(device);
        auto vbuf = buffers[VERTEX_BUFFER].buf;
        vkCmdBindVertexBuffers(cmdbuf, 0, 1, &vbuf, &offset);
        vkCmdBindIndexBuffer(cmdbuf, buffers[INDEX_BUFFER].buf, 0, VK_INDEX_TYPE_UINT32);
    }

    void ReleaseDeviceData(VkDevice device)
    {
        for(auto& buffer : buffers_by_device.at(device)) {
            if(buffer.mapped) {
                vkUnmapMemory(device, buffer.mem);
            }
            vkDestroyBuffer(device, buffer.buf, nullptr);
            vkFreeMemory(device, buffer.mem, nullptr);
        }

        buffers_by_device.erase(device);
    }
};

std::tuple<VkImage, VkDeviceMemory> CreateBound2DImage(VkPhysicalDevice physical_device, VkDevice device, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage_flags, VkImageLayout initial_layout, VkMemoryPropertyFlags properties)
{
    VkImageCreateInfo create_image {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent{width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage_flags,
        .initialLayout = initial_layout,
    };
    VkImage image;
    VK_CHECK(vkCreateImage(device, &create_image, nullptr, &image));

    VkMemoryRequirements imageMemReqs;
    vkGetImageMemoryRequirements(device, image, &imageMemReqs);
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    uint32_t memoryTypeIndex = GetMemoryTypeIndex(memory_properties, imageMemReqs.memoryTypeBits, properties);
    VkMemoryAllocateInfo allocate {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = imageMemReqs.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    VkDeviceMemory image_memory;
    VK_CHECK(vkAllocateMemory(device, &allocate, nullptr, &image_memory));
    VK_CHECK(vkBindImageMemory(device, image, image_memory, 0));
    return {image, image_memory};
}

VkSwapchainKHR CreateSwapchain(VkDevice device, VkSurfaceKHR surface, int32_t min_image_count, VkFormat chosen_color_format, VkColorSpaceKHR chosen_color_space, VkPresentModeKHR swapchain_present_mode, uint32_t width, uint32_t height)
{
    // XXX verify present mode with vkGetPhysicalDeviceSurfacePresentModesKHR

    VkSwapchainCreateInfoKHR create {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = static_cast<uint32_t>(min_image_count),
        .imageFormat = chosen_color_format,
        .imageColorSpace = chosen_color_space,
        .imageExtent { width, height },
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = swapchain_present_mode,
        .clipped = true,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    VkSwapchainKHR swapchain;
    VK_CHECK(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain));
    return swapchain;
}


std::vector<VkImage> GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain)
{
    uint32_t swapchain_image_count;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr));
    std::vector<VkImage> swapchain_images(swapchain_image_count);
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data()));
    return swapchain_images;
}


namespace VulkanApp
{

bool be_verbose = true;
bool enable_validation = false;

// non-frame stuff - instance, queue, device, etc
VkInstance instance;
VkPhysicalDevice physical_device;
VkDevice device;
uint32_t graphics_queue_family = NO_QUEUE_FAMILY;
VkSurfaceKHR surface;
VkSwapchainKHR swapchain;
VkCommandPool command_pool;
VkQueue queue;
std::vector<UniformBuffer> uniforms;
std::vector<ImageSampler> samplers;

// In flight rendering stuff
int submission_index = 0;
struct Submission {
    VkCommandBuffer command_buffer { VK_NULL_HANDLE };
    bool draw_completed_fence_submitted { false };
    VkFence draw_completed_fence { VK_NULL_HANDLE };
    VkSemaphore draw_completed_semaphore { VK_NULL_HANDLE };
    VkDescriptorSet descriptor_set { VK_NULL_HANDLE };
    Buffer uniform_buffers[3];
};
static constexpr int SUBMISSIONS_IN_FLIGHT = 2;
std::vector<Submission> submissions(SUBMISSIONS_IN_FLIGHT);

// per-frame stuff - swapchain image, current layout, indices, fences, semaphores
VkSurfaceFormatKHR chosen_surface_format;
VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;
VkFormat chosen_color_format;
VkFormat chosen_depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
uint32_t swapchain_image_count = 3;
struct PerSwapchainImage {
    VkImageLayout layout;
    VkImage image;
    VkImageView image_view;
    VkFramebuffer framebuffer;
};
std::vector<PerSwapchainImage> per_swapchainimage;
uint32_t swapchainimage_semaphore_index = 0;
std::vector<VkSemaphore> swapchainimage_semaphores;
VkImage depth_image;
VkDeviceMemory depth_image_memory;
VkImageView depth_image_view;

// rendering stuff - pipelines, binding & drawing commands
VkPipelineLayout pipeline_layout;
VkDescriptorPool descriptor_pool;
VkRenderPass renderPass;
VkPipeline pipeline;
VkDescriptorSetLayout descriptor_set_layout;

// interaction data
enum { DRAW_VULKAN, DRAW_CPU, DRAW_MODE_COUNT };
int drawing_mode = DRAW_CPU;
std::map<int, std::string> DrawingModeNames = {
    {DRAW_VULKAN, "Vulkan"},
    {DRAW_CPU, "CPU"},
};

float frame = 0.0;

aabox volume_bounds;
manipulator volume_manip;
manipulator object_manip;
manipulator light_manip;
manipulator* current_manip;
int buttonPressed = -1;
bool motionReported = true;
double oldMouseX;
double oldMouseY;
float fov = 45;

// geometry data
typedef std::unique_ptr<Drawable> DrawablePtr;
DrawablePtr drawable;

void InitializeInstance()
{
    if (be_verbose) {
        PrintImplementationInformation();
    }
    instance = CreateInstance(enable_validation);
}

void CreatePerSubmissionData()
{
    for(uint32_t i = 0; i < SUBMISSIONS_IN_FLIGHT; i++) {

        auto& submission = submissions[i];

        const VkCommandBufferAllocateInfo allocate {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(device, &allocate, &submission.command_buffer));

        VkFenceCreateInfo fence_create {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = 0,
        };
        VK_CHECK(vkCreateFence(device, &fence_create, nullptr, &submission.draw_completed_fence));
        submission.draw_completed_fence_submitted = false;

        VkSemaphoreCreateInfo sema_create {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
        };
        VK_CHECK(vkCreateSemaphore(device, &sema_create, NULL, &submission.draw_completed_semaphore));

        VkDescriptorSetAllocateInfo allocate_descriptor_set {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &allocate_descriptor_set, &submission.descriptor_set));

        int which = 0;
        for(const auto& uniform: uniforms) {
            auto &uniform_buffer = submission.uniform_buffers[which];

            uniform_buffer.Create(physical_device, device, uniform.size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VK_CHECK(vkMapMemory(device, uniform_buffer.mem, 0, uniform.size, 0, &uniform_buffer.mapped));

            VkDescriptorBufferInfo buffer_info { uniform_buffer.buf, 0, uniform.size };
            VkWriteDescriptorSet write_descriptor_set {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = submission.descriptor_set,
                .dstBinding = uniform.binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &buffer_info,
                .pTexelBufferView = nullptr,
            };
            vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);

            which++;
        }

        if(samplers.size() != 1) {
            // XXX only one texture and one drawable at the moment
            throw std::runtime_error("More than one sampler is not yet supported");
        }
        for(const auto& sampler: samplers) {
            VkDescriptorImageInfo image_info {
                .sampler = drawable->textureSampler,
                .imageView = drawable->textureImageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            VkWriteDescriptorSet write_descriptor_set {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = submission.descriptor_set,
                .dstBinding = sampler.binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_info,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            };
            vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);
        }
    }
}

void WaitForAllDrawsCompleted()
{
    for(auto& submission: submissions) {
        if(submission.draw_completed_fence_submitted) {
            VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
            VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
            submission.draw_completed_fence_submitted = false;
        }
    }
}

void DestroySwapchainData(VkDevice device)
{
    WaitForAllDrawsCompleted();

    for(auto& sema: swapchainimage_semaphores) {
        vkDestroySemaphore(device, sema, nullptr);
        sema = VK_NULL_HANDLE;
    }
    swapchainimage_semaphores.clear();

    vkDestroyImageView(device, depth_image_view, nullptr);
    depth_image_view = VK_NULL_HANDLE;
    vkDestroyImage(device, depth_image, nullptr);
    depth_image = VK_NULL_HANDLE;
    vkFreeMemory(device, depth_image_memory, nullptr);
    depth_image_memory = VK_NULL_HANDLE;

    for(uint32_t i = 0; i < swapchain_image_count; i++) {
        auto& per_image = per_swapchainimage[i];
        vkDestroyImageView(device, per_image.image_view, nullptr);
        per_image.image_view = VK_NULL_HANDLE;
        vkDestroyFramebuffer(device, per_image.framebuffer, nullptr);
        per_image.framebuffer = VK_NULL_HANDLE;
    }
    per_swapchainimage.clear();

    vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
}

void CreateSwapchainData(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface, uint32_t width, uint32_t height, VkRenderPass render_pass)
{
    swapchain = CreateSwapchain(device, surface, swapchain_image_count, chosen_color_format, chosen_surface_format.colorSpace, swapchain_present_mode, width, height);

// frame-related stuff - swapchains indices, fences, semaphores

    auto [depth_image, depth_image_memory] = CreateBound2DImage(physical_device, device, chosen_depth_format, width, height, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    depth_image_view = CreateImageView(device, chosen_depth_format, depth_image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    std::vector<VkImage> swapchain_images = GetSwapchainImages(device, swapchain);
    assert(swapchain_image_count == swapchain_images.size());
    swapchain_image_count = static_cast<uint32_t>(swapchain_images.size());

    per_swapchainimage.resize(swapchain_image_count);
    for(uint32_t i = 0; i < swapchain_image_count; i++) {
        auto& per_image = per_swapchainimage[i];

        per_image.image = swapchain_images[i];
        per_image.layout = VK_IMAGE_LAYOUT_UNDEFINED;

        per_image.image_view = CreateImageView(device, chosen_color_format, per_swapchainimage[i].image, VK_IMAGE_ASPECT_COLOR_BIT);

        per_image.framebuffer = CreateFramebuffer(device, {per_image.image_view, depth_image_view}, render_pass, width, height);
        VkImageView imageviews[] = {per_image.image_view, depth_image_view};
        VkFramebufferCreateInfo framebufferCreate {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = static_cast<uint32_t>(std::size(imageviews)),
            .pAttachments = imageviews,
            .width = width,
            .height = height,
            .layers = 1,
        };
        VK_CHECK(vkCreateFramebuffer(device, &framebufferCreate, nullptr, &per_image.framebuffer));
    }

    swapchainimage_semaphores.resize(swapchain_image_count);
    for(uint32_t i = 0; i < swapchain_image_count; i++) {
        // XXX create a timeline semaphore by chaining after a
        // VkSemaphoreTypeCreateInfo with VkSemaphoreTypeCreateInfo =
        // VK_SEMAPHORE_TYPE_TIMELINE
        VkSemaphoreCreateInfo sema_create {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
        };
        VK_CHECK(vkCreateSemaphore(device, &sema_create, NULL, &swapchainimage_semaphores[i]));

    }
}

void InitializeState()
{
    // non-frame stuff
    physical_device = ChoosePhysicalDevice(instance);

    uint32_t formatCount;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, surfaceFormats.data()));

    VkSurfaceCapabilitiesKHR surfcaps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surfcaps));

    chosen_surface_format = PickSurfaceFormat(surfaceFormats);
    chosen_color_format = chosen_surface_format.format;

    graphics_queue_family = FindQueueFamily(physical_device, VK_QUEUE_GRAPHICS_BIT);
    if(graphics_queue_family == NO_QUEUE_FAMILY) {
        fprintf(stderr, "couldn't find a graphics queue\n");
        abort();
    }

    if(be_verbose) {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        PrintDeviceInformation(physical_device, memory_properties);
    }

    std::vector<const char*> device_extensions;

    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

#ifdef PLATFORM_MACOS
    device_extensions.push_back("VK_KHR_portability_subset" /* VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME */);
#endif

#if 0
    device_extensions.insert(extensions.end(), {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME
        });
#endif

    if (be_verbose) {
        for (const auto& e : device_extensions) {
            printf("asked for %s\n", e);
        }
    }

    device = CreateDevice(physical_device, device_extensions, graphics_queue_family);

    vkGetDeviceQueue(device, graphics_queue_family, 0, &queue);

    VkCommandPoolCreateInfo create_command_pool {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_queue_family,
    };
    VK_CHECK(vkCreateCommandPool(device, &create_command_pool, nullptr, &command_pool));

    // XXX Should probe these from shader code somehow
    // XXX at the moment this order is assumed for "struct submission"
    // uniforms Buffer structs, see *_uniforms setting code in DrawFrame
    uniforms.push_back({0, VK_SHADER_STAGE_VERTEX_BIT, sizeof(VertexUniforms)});
    uniforms.push_back({1, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(FragmentUniforms)});
    uniforms.push_back({2, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ShadingUniforms)});
    samplers.push_back({3, VK_SHADER_STAGE_FRAGMENT_BIT});

    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    for(const auto& uniform: uniforms) {
        VkDescriptorSetLayoutBinding binding {
            .binding = uniform.binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = uniform.stageFlags,
            .pImmutableSamplers = nullptr,
        };
        layout_bindings.push_back(binding);
    }
    for(const auto& sampler: samplers) {
        VkDescriptorSetLayoutBinding binding {
            .binding = sampler.binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = sampler.stageFlags,
            .pImmutableSamplers = nullptr,
        };
        layout_bindings.push_back(binding);
    }

    VkDescriptorPoolSize pool_sizes[] {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(uniforms.size()) * SUBMISSIONS_IN_FLIGHT },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(samplers.size()) * SUBMISSIONS_IN_FLIGHT },
    };
    VkDescriptorPoolCreateInfo create_descriptor_pool {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,
        .maxSets = SUBMISSIONS_IN_FLIGHT,
        .poolSizeCount = std::size(pool_sizes),
        .pPoolSizes = pool_sizes,
    };
    VK_CHECK(vkCreateDescriptorPool(device, &create_descriptor_pool, nullptr, &descriptor_pool));

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = 0,
        .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
        .pBindings = layout_bindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create, nullptr, &descriptor_set_layout));

    drawable->CreateDeviceData(physical_device, device, queue);

    CreatePerSubmissionData();

// rendering stuff - pipelines, binding & drawing commands

    VkAttachmentDescription color_attachment_description {
        .flags = 0,
        .format = chosen_color_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    VkAttachmentDescription depth_attachment_description {
        .flags = 0,
        .format = chosen_depth_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentDescription attachments[] = { color_attachment_description, depth_attachment_description };

    VkAttachmentReference colorReference { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthReference { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass {
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorReference,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = &depthReference,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };

    VkSubpassDependency color_attachment_dependency {
        // Image Layout Transition
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
        .dependencyFlags = 0,
    };
    VkSubpassDependency depth_attachment_dependency {
        // Depth buffer is shared between swapchain images
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };
    VkSubpassDependency attachment_dependencies[] = {
        depth_attachment_dependency,
        color_attachment_dependency,
    };

    VkRenderPassCreateInfo render_pass_create {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .flags = 0,
        .attachmentCount = static_cast<uint32_t>(std::size(attachments)),
        .pAttachments = attachments,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(std::size(attachment_dependencies)),
        .pDependencies = attachment_dependencies,
    };
    VK_CHECK(vkCreateRenderPass(device, &render_pass_create, nullptr, &renderPass));

    // Swapchain and per-swapchainimage stuff
    // Creating the framebuffer requires the renderPass
    CreateSwapchainData(physical_device, device, surface, surfcaps.currentExtent.width, surfcaps.currentExtent.height, renderPass);

    // Create a graphics pipeline
    VkVertexInputBindingDescription vertex_input_binding {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = Vertex::GetVertexInputAttributeDescription();

    VkPipelineVertexInputStateCreateInfo vertex_input_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_input_attributes.size()),
        .pVertexAttributeDescriptions = vertex_input_attributes.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE, // VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    VkPipelineColorBlendAttachmentState att_state[] {
        {
            .blendEnable = VK_FALSE,
            .colorWriteMask = 0xf,
        },
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(std::size(att_state)),
        .pAttachments = att_state,
    };

    VkStencilOpState keep_always{ .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = keep_always,
        .back = keep_always,
    };

    VkPipelineViewportStateCreateInfo viewport_state { 
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .pSampleMask = NULL,
    };

    VkDynamicState dynamicStateEnables[]{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamicState {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStateEnables)),
        .pDynamicStates = dynamicStateEnables,
    };

    VkPipelineLayoutCreateInfo create_layout {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &create_layout, nullptr, &pipeline_layout));

    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    std::vector<std::pair<std::string, VkShaderStageFlagBits>> shader_binaries {
        {"volume.vert", VK_SHADER_STAGE_VERTEX_BIT},
        {"volume.frag", VK_SHADER_STAGE_FRAGMENT_BIT}
    };
    
    for(const auto& [name, stage]: shader_binaries) {
        std::vector<uint32_t> shader_code = GetFileAsCode(name);
        VkShaderModule shader_module = CreateShaderModule(device, shader_code);
        VkPipelineShaderStageCreateInfo shader_create {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = stage,
            .module = shader_module,
            .pName = "main",
        };
        shader_stages.push_back(shader_create);
    }

    VkGraphicsPipelineCreateInfo create_pipeline {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = static_cast<uint32_t>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = &depth_stencil_state,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = &dynamicState,
        .layout = pipeline_layout,
        .renderPass = renderPass,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_pipeline, nullptr, &pipeline));
}

void Cleanup()
{
    WaitForAllDrawsCompleted();
    drawable->ReleaseDeviceData(device);
}

struct ImageDataWrapper
{
    std::vector<uint8_t> &image_data;
    int width;
    int height;
    ImageDataWrapper(std::vector<uint8_t> &image_data, int width, int height) :
        image_data(image_data), width(width), height(height) 
    {}
    VkFormat GetVulkanFormat() { return VK_FORMAT_R8G8B8A8_UNORM; }
    int GetWidth() { return width; }
    int GetHeight() { return height; }
    void* GetData() { return image_data.data(); }
    size_t GetSize() { return width * height * 4; }
};

std::vector<uint16_t> ct_data;
std::vector<vec3> ct_data_normal;

std::tuple<int, int, int> NormalizedToIndices(const vec3& uvw)
{
    int i = std::clamp(static_cast<int>(uvw[0] * 512), 0, 511);
    int j = std::clamp(static_cast<int>(uvw[1] * 512), 0, 511);
    int k = std::clamp(static_cast<int>(uvw[2] * 114), 0, 113);
    return {i, j, k};
}

uint16_t LookupCTDensity(int i, int j, int k)
{
    return ct_data[i + j * 512 + k * 512 * 512];
}

uint16_t LookupCTDensity(const vec3& uvw)
{
    auto [i, j, k] = NormalizedToIndices(uvw);
    return LookupCTDensity(i, j, k);
}

void SetCTNormal(const vec3& uvw, const vec3& normal)
{
    auto [i, j, k] = NormalizedToIndices(uvw);
    ct_data_normal[i + j * 512 + k * 512 * 512] = normal;
}

vec3 LookupCTNormal(int i, int j, int k)
{
    return ct_data_normal[i + j * 512 + k * 512 * 512];
}

vec3 LookupCTNormal(const vec3& uvw)
{
    auto [i, j, k] = NormalizedToIndices(uvw);
    return LookupCTNormal(i, j, k);
}

void CalculateGradients()
{
    float du = 1.1f / 512;
    float dv = 1.1f / 512;
    float dw = 1.1f / 114;

    for(int k = 0; k < 114; k++) {
        for(int j = 0; j < 512; j++) {
            for(int i = 0; i < 512; i++) {
                vec3 uvw { i / 512.0f, j / 512.0f, k / 114.0f };
                float v = LookupCTDensity(uvw);
                float gu = (LookupCTDensity(uvw + vec3(du, 0, 0)) - v) / du;
                float gv = (LookupCTDensity(uvw + vec3(0, dv, 0)) - v) / dv;
                float gw = (LookupCTDensity(uvw + vec3(0, 0, dw)) - v) / dw;
                vec3 normal {gu, gv, gw};
                if(length(normal) > .001) {
                    SetCTNormal(uvw, normalize(normal));
                } else {
                    SetCTNormal(uvw, vec3(0, 0, 0));
                }
            }
        }
    }
}

void LoadCTData()
{
    ct_data.resize(512 * 512 * 114);
    ct_data_normal.resize(512 * 512 * 114);
    if(true) {
        for(int i = 0; i < 114; i++) {
            static char filename[512];
            snprintf(filename, sizeof(filename), "%s/file_%03d.bin", getenv("IMAGES"), i);
            FILE *fp = fopen(filename, "rb");
            if(!fp) {
                printf("couldn't open \"%s\" for reading\n", filename);
                exit(EXIT_FAILURE);
            }
            for(int row = 0; row < 512; row++) {
                static uint16_t rowbuffer[512];
                size_t was_read = fread(rowbuffer, 2, 512, fp);
                if(was_read != 512) {
                    printf("short read from \"%s\"\n", filename);
                    exit(EXIT_FAILURE);
                }
                for(int column = 0; column < 512; column++) {
                    ct_data[512 * 512 * i + 512 * row + column] = rowbuffer[512 - 1 - column];
                }
            }
            fclose(fp);

#if 0
            snprintf(filename, sizeof(filename), "file_%03d.ppm", i);
            fp = fopen(filename, "wb");
            if(!fp) {
                printf("couldn't open \"%s\" for writing\n", filename);
                exit(EXIT_FAILURE);
            }
            fprintf(fp, "P5 512 512 255\n");
            for(int j = 0; j < 512 * 512; j++) {
                uint8_t b = ct_data[j + i * 512 * 512] % 256;
                fwrite(&b, 1, 1, fp);
            }
            fclose(fp);
#endif
        }
    } else {
        for(int k = 0; k < 114; k++) {
            for(int j = 0; j < 512; j++) {
                for(int i = 0; i < 512; i++) {
                    vec3 uvw { i / 512.0f, j / 512.0f, k / 114.0f };
                    float r = length(uvw - vec3(.5f, .5f, .5f));
                    int index = i + j * 512 + k * 512 * 512;
                    ct_data[index] = (r > .5) ? 0 : 65535;
                }
            }
        }
    }
    CalculateGradients();
}

uint16_t threshold = 8500;

bool trace_volume(const ray& ray, vec3& color, vec3& normal)
{
    range ray_range{0, std::numeric_limits<float>::max()};
    range rn = ray_range.intersect(ray_intersect_box(volume_bounds, ray));
    if(rn) {
        // vec3 enter_volume = ray.at(rn.t0);
        // vec3 exit_volume = ray.at(rn.t1);
        for(float t = rn.t0; t < rn.t1; t += 1/512.0) {
            // enter_volume -= volume_bounds.boxmin;
            // exit_volume -= volume_bounds.boxmin;
            uint16_t density = LookupCTDensity(ray.at(t));
            if(density > threshold) {
                normal = LookupCTNormal(ray.at(t));
                if(density > 8300) {
                    color = vec3(1, 1, 1); 
                } else {
                    color = vec3(.8f, .2f, .2f); 
                }
                return true;
            }
        }
    }
    return false;
}

void Render(uint8_t *image_data, int surfWidth, int surfHeight)
{
    mat4f modelview = volume_manip.m_matrix;
    mat4f modelview_3x3 = modelview;
    modelview_3x3.m_v[12] = 0.0f; modelview_3x3.m_v[13] = 0.0f; modelview_3x3.m_v[14] = 0.0f;
    mat4f modelview_normal = inverse(transpose(modelview_3x3));

    static constexpr int tileWidth = 32;
    static constexpr int tileHeight = 32;

    std::vector<std::tuple<int, int, int, int>> tiles;

    for(int tileY = 0; tileY < surfHeight; tileY += tileHeight) {
        for(int tileX = 0; tileX < surfWidth; tileX += tileWidth) {
            tiles.push_back({tileX, tileY, std::min(tileX + tileWidth, surfWidth), std::min(tileY + tileHeight, surfHeight)});
        }
    }

    auto renderTile = [=](const std::tuple<int, int, int, int>& bounds) {
        auto [tileX, tileY, width, height] = bounds;
        for(int y = tileY; y < height; y ++) {
            for(int x = tileX; x < width; x ++) {
                float u = ((x + .5f) / (float)surfWidth) * 2.0f - 1.0f;
                float v = ((surfHeight - (y + .5f) - 1) / (float)surfHeight) * 2.0f - 1.0f;

                vec3 o {0, 0, 0};
                vec3 d {u, v, -1};
                ray eye_ray {{0, 0, 0}, {u, v, -1}};

                mat4f to_object = inverse(volume_manip.m_matrix);
                ray object_ray = eye_ray * to_object;

                vec3 surface_color;
                vec3 surface_normal;
                bool hit = trace_volume(object_ray, surface_color, surface_normal);
                vec3 normal = surface_normal * modelview_normal;

                vec3 color;
                if(hit && (normal[0] != 0.0f) && (normal[1] != 0.0f) && (normal[2] != 0.0f)) {
                    // float lighting = fabsf(dot(normal, vec3(.577f, .577f, .577f)));
                    float lighting = fabsf(dot(normal, vec3(0, 0, 1)));
                    color = surface_color * lighting;
                } else {
                    color = vec3(0, 0, 0);
                }

                int index = (x + y * surfWidth) * 4;
                image_data[index + 2] = 255 * std::clamp(color[0], 0.0f, 1.0f);
                image_data[index + 1] = 255 * std::clamp(color[1], 0.0f, 1.0f);
                image_data[index + 0] = 255 * std::clamp(color[2], 0.0f, 1.0f);
            }
        }
    };

    // std::for_each(std::parallel_unsequenced_policy, tiles.begin(), tiles.end(), renderTile);
    std::vector<std::thread*> threads;
    for(auto const& t: tiles) {
        auto f = [=]() {
            renderTile(t);
        };
        threads.push_back(new std::thread(f));
    }
    while(!threads.empty()) {
        std::thread* thread = threads.back();
        threads.pop_back();
        thread->join();
        delete thread;
    }
}

void DrawFrameCPU([[maybe_unused]] GLFWwindow *window)
{
    VkSurfaceCapabilitiesKHR surfcaps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surfcaps));
    int32_t surfWidth = surfcaps.currentExtent.width;
    int32_t surfHeight = surfcaps.currentExtent.height;

    static Buffer staging_buffer;
    static size_t previous_size = 0;
    size_t image_size = surfWidth * surfHeight * 4;
    if(image_size > previous_size) {
        staging_buffer.Create(physical_device, device, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkMapMemory(device, staging_buffer.mem, 0, image_size, 0, &staging_buffer.mapped));
        previous_size = image_size;
    }

    uint8_t* image_data = static_cast<uint8_t*>(staging_buffer.mapped);
    Render(image_data, surfWidth, surfHeight);
    auto& submission = submissions[submission_index];

    if(submission.draw_completed_fence_submitted) {
        VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
        submission.draw_completed_fence_submitted = false;
    }

    VkResult result;
    uint32_t swapchain_index;
    while((result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainimage_semaphores[swapchainimage_semaphore_index], VK_NULL_HANDLE, &swapchain_index)) != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData(device);
            CreateSwapchainData(physical_device, device, surface, surfWidth, surfHeight, renderPass);
        } else {
	    std::cerr << "VkResult from vkAcquireNextImageKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }
    auto& per_image = per_swapchainimage[swapchain_index];

    auto cb = submission.command_buffer;

    VkCommandBufferBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));

    VkImageMemoryBarrier transfer_dst_optimal {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .oldLayout = per_image.layout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image = per_image.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transfer_dst_optimal);

    // Copy buffer to image
    VkBufferImageCopy copy {
        .bufferOffset = 0,
        .bufferRowLength = 0, // static_cast<uint32_t>(surfWidth),
        .bufferImageHeight = 0, // static_cast<uint32_t>(surfHeight),
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {static_cast<uint32_t>(surfWidth), static_cast<uint32_t>(surfHeight), 1},
    };

    vkCmdCopyBufferToImage(cb, staging_buffer.buf, per_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier present_src {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .image = per_image.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &present_src);

    VK_CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags waitdststagemask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchainimage_semaphores[swapchainimage_semaphore_index],
        .pWaitDstStageMask = &waitdststagemask,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &submission.draw_completed_semaphore,
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, submission.draw_completed_fence));
    submission.draw_completed_fence_submitted = true;

    // 13. Present the rendered result
    VkPresentInfoKHR present {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &submission.draw_completed_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_index,
        .pResults = nullptr,
    };
    result = vkQueuePresentKHR(queue, &present);
    if(result != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData(device);
            CreateSwapchainData(physical_device, device, surface, surfWidth, surfHeight, renderPass);
        } else {
	    std::cerr << "VkResult from vkQueuePresentKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }

    VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
    VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
    submission.draw_completed_fence_submitted = false;

    per_image.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    submission_index = (submission_index + 1) % submissions.size();
    swapchainimage_semaphore_index = (swapchainimage_semaphore_index + 1) % swapchainimage_semaphores.size();

    frame += 1;
}

void DrawFrameVulkan([[maybe_unused]] GLFWwindow *window)
{
    VkSurfaceCapabilitiesKHR surfcaps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surfcaps));

    auto& submission = submissions[submission_index];

    if(submission.draw_completed_fence_submitted) {
        VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
        submission.draw_completed_fence_submitted = false;
    }

    mat4f modelview = object_manip.m_matrix;
    mat4f modelview_3x3 = modelview;
    modelview_3x3.m_v[12] = 0.0f; modelview_3x3.m_v[13] = 0.0f; modelview_3x3.m_v[14] = 0.0f;
    mat4f modelview_normal = inverse(transpose(modelview_3x3));

    float nearClip = .1f; // XXX - gSceneManip->m_translation[2] - gSceneManip->m_reference_size;
    float farClip = 1000.0; // XXX - gSceneManip->m_translation[2] + gSceneManip->m_reference_size;
    float frustumTop = tan(fov / 180.0f * 3.14159f / 2) * nearClip;
    float frustumBottom = -frustumTop;
    float frustumRight = frustumTop * surfcaps.currentExtent.width / surfcaps.currentExtent.height;
    float frustumLeft = -frustumRight;
    mat4f projection = mat4f::frustum(frustumLeft, frustumRight, frustumTop, frustumBottom, nearClip, farClip);

    VertexUniforms* vertex_uniforms = static_cast<VertexUniforms*>(submission.uniform_buffers[0].mapped);
    vertex_uniforms->modelview = modelview;
    vertex_uniforms->modelview_normal = modelview_normal;
    vertex_uniforms->projection = projection.m_v;

    vec4 light_position{1000, 1000, 1000, 0};
    vec3 light_color{1, 1, 1};

    light_position = light_position * light_manip.m_matrix;

    FragmentUniforms* fragment_uniforms = static_cast<FragmentUniforms*>(submission.uniform_buffers[1].mapped);
    fragment_uniforms->light_position[0] = light_position[0];
    fragment_uniforms->light_position[1] = light_position[1];
    fragment_uniforms->light_position[2] = light_position[2];
    fragment_uniforms->light_color = light_color;

    ShadingUniforms* shading_uniforms = static_cast<ShadingUniforms*>(submission.uniform_buffers[2].mapped);
    shading_uniforms->specular_color.set(drawable->specular_color); // XXX drops specular_color[3]
    shading_uniforms->shininess = drawable->shininess;

    VkResult result;
    uint32_t swapchain_index;
    while((result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainimage_semaphores[swapchainimage_semaphore_index], VK_NULL_HANDLE, &swapchain_index)) != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData(device);
            CreateSwapchainData(physical_device, device, surface, surfcaps.currentExtent.width, surfcaps.currentExtent.height, renderPass);
        } else {
	    std::cerr << "VkResult from vkAcquireNextImageKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }
    auto& per_image = per_swapchainimage[swapchain_index];

    auto cb = submission.command_buffer;

    VkCommandBufferBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    const VkClearValue clearValues [2] {
        {.color {.float32 {0.1f, 0.1f, 0.2f, 1.0f}}},
        {.depthStencil = {1.0f, 0}},
    };
    VkRenderPassBeginInfo beginRenderpass {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = renderPass,
        .framebuffer = per_image.framebuffer,
        .renderArea = {{0, 0}, {surfcaps.currentExtent.width, surfcaps.currentExtent.height}},
        .clearValueCount = static_cast<uint32_t>(std::size(clearValues)),
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(cb, &beginRenderpass, VK_SUBPASS_CONTENTS_INLINE);

    // 6. Bind the graphics pipeline state
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &submission.descriptor_set, 0, NULL);

    // 9. Set viewport and scissor parameters
    VkViewport viewport {
        .x = 0,
        .y = 0,
        .width = (float)surfcaps.currentExtent.width,
        .height = (float)surfcaps.currentExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor {
        .offset{0, 0},
        .extent{surfcaps.currentExtent.width, surfcaps.currentExtent.height}};
    vkCmdSetScissor(cb, 0, 1, &scissor);

    drawable->BindForDraw(device, cb);
    vkCmdDrawIndexed(cb, drawable->triangleCount * 3, 1, 0, 0, 0);

    vkCmdEndRenderPass(cb);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags waitdststagemask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchainimage_semaphores[swapchainimage_semaphore_index],
        .pWaitDstStageMask = &waitdststagemask,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &submission.draw_completed_semaphore,
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, submission.draw_completed_fence));
    submission.draw_completed_fence_submitted = true;

    // 13. Present the rendered result
    VkPresentInfoKHR present {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &submission.draw_completed_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_index,
        .pResults = nullptr,
    };
    result = vkQueuePresentKHR(queue, &present);
    if(result != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData(device);
            CreateSwapchainData(physical_device, device, surface, surfcaps.currentExtent.width, surfcaps.currentExtent.height, renderPass);
        } else {
	    std::cerr << "VkResult from vkQueuePresentKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }

    submission_index = (submission_index + 1) % submissions.size();
    swapchainimage_semaphore_index = (swapchainimage_semaphore_index + 1) % swapchainimage_semaphores.size();

    frame += 1;
}

void DrawFrame(GLFWwindow *window)
{
    if(drawing_mode == DRAW_VULKAN) {
        DrawFrameVulkan(window);
    } else if (drawing_mode == DRAW_CPU) {
        DrawFrameCPU(window);
    }
}

};

static void ErrorCallback([[maybe_unused]] int error, const char* description)
{
    fprintf(stderr, "GLFW: %s\n", description);
}

static void KeyCallback(GLFWwindow *window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods)
{
    using namespace VulkanApp;

    if(action == GLFW_PRESS) {
        switch(key) {
            case 'M':
                if(false) {
                    drawing_mode = (drawing_mode + 1) % DRAW_MODE_COUNT;
                    printf("Current drawing mode: %s\n", DrawingModeNames[drawing_mode].c_str());
                    if(drawing_mode == DRAW_VULKAN) {
                        current_manip = &object_manip;
                    } else {
                        current_manip = &volume_manip;
                        volume_manip.m_mode = manipulator::ROTATE;
                    }
                }
                break;

            case 'W':
                break;

            case 'R':
                current_manip = &volume_manip;
                volume_manip.m_mode = manipulator::ROTATE;
                break;

            case 'O':
                current_manip = &volume_manip;
                volume_manip.m_mode = manipulator::ROLL;
                break;

            case 'X':
                current_manip = &volume_manip;
                volume_manip.m_mode = manipulator::SCROLL;
                break;

            case 'Z':
                current_manip = &volume_manip;
                volume_manip.m_mode = manipulator::DOLLY;
                break;

            case 'L':
                current_manip = &light_manip;
                light_manip.m_mode = manipulator::ROTATE;
                break;

            case 'Q': case GLFW_KEY_ESCAPE: case '\033':
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;

            case GLFW_KEY_LEFT_BRACKET:
                threshold -= 1000;
                printf("threshold %d\n", threshold);
                break;

            case GLFW_KEY_RIGHT_BRACKET:
                threshold += 1000;
                printf("threshold %d\n", threshold);
                break;

            case GLFW_KEY_SEMICOLON:
                threshold -= 100;
                printf("threshold %d\n", threshold);
                break;

            case GLFW_KEY_APOSTROPHE:
                threshold += 100;
                printf("threshold %d\n", threshold);
                break;

            case GLFW_KEY_COMMA:
                threshold -= 10;
                printf("threshold %d\n", threshold);
                break;

            case GLFW_KEY_PERIOD:
                threshold += 10;
                printf("threshold %d\n", threshold);
                break;
        }
    }
}

static void ButtonCallback(GLFWwindow *window, int b, int action, [[maybe_unused]] int mods)
{
    using namespace VulkanApp;

    double x, y;
    glfwGetCursorPos(window, &x, &y);

    if(b == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS) {
        buttonPressed = 1;
	oldMouseX = x;
	oldMouseY = y;
    } else {
        buttonPressed = -1;
    }
}

static void MotionCallback(GLFWwindow *window, double x, double y)
{
    using namespace VulkanApp;

    // glfw/glfw#103
    // If no motion has been reported yet, we catch the first motion
    // reported and store the current location
    if(!motionReported) {
        motionReported = true;
        oldMouseX = x;
        oldMouseY = y;
    }

    double dx, dy;

    dx = x - oldMouseX;
    dy = y - oldMouseY;

    oldMouseX = x;
    oldMouseY = y;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    if(buttonPressed == 1) {
        current_manip->move(static_cast<float>(dx / width), static_cast<float>(dy / height));
    }
}

static void ScrollCallback(GLFWwindow *window, double dx, double dy)
{
    using namespace VulkanApp;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    current_manip->move(static_cast<float>(dx / width), static_cast<float>(dy / height));
}

bool ParseTriSrc(FILE *fp, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::string& texture_name, float specular_color[4], float& shininess)
{
    char texture_name_c_str[512];
    char tag_name_c_str[512];

    // XXX each triangle can reference a texture, but I ignore all but the last in this hacked parser
    while(fscanf(fp,"\"%[^\"]\"", texture_name_c_str) == 1) {
	if(fscanf(fp,"%s ", tag_name_c_str) != 1) {
            // XXX each triangle can reference a tag name, but I ignore all but the last in this hacked parser
            // It really can be ignored in a sense because it's a hint about part assemblies, spatial locality, etc
	    fprintf(stderr, "couldn't read tag name\n");
	    return false;
	}

        // XXX each triangle can reference object specular and shininess, but I ignore all but the last in this hacked parser
	if(fscanf(fp,"%f %f %f %f %f ", &specular_color[0], &specular_color[1],
	    &specular_color[2], &specular_color[3], &shininess) != 5) {
	    fprintf(stderr, "couldn't read specular properties\n");
	    return false;
	}

	if(shininess > 0 && shininess < 1) {
	    // XXX I forgot what shininess units are!
	    shininess *= 10;
	}

        for(int i = 0; i < 3; i++) {
            float v[3];
            float n[3];
            float c[4];
            float t[2];

	    if(fscanf(fp,"%g %g %g %g %g %g %g %g %g %g %g %g ",
	        &v[0], &v[1], &v[2],
	        &n[0], &n[1], &n[2],
	        &c[0], &c[1], &c[2], &c[3],
	        &t[0], &t[1]) != 12) {

		fprintf(stderr, "couldn't read Vertex\n");
		return false;
	    }
            indices.push_back(static_cast<uint32_t>(vertices.size()));
            vertices.push_back(Vertex(v, n, c, t));
        }

        // old code from another project for reference
        // MATERIAL mtl(texture_name, specular_color, shininess);
        // sets.get_triangle_set(tag_name, mtl).add_triangle(verts[0], verts[1], verts[2]);
    }

    texture_name = texture_name_c_str;

    return true;
}


// very old code
static void skipComments(FILE *fp)
{
    int c;

    while((c = fgetc(fp)) == '#')
        while((c = fgetc(fp)) != '\n');
    ungetc(c, fp);
}


int pnmRead(FILE *file, int *w, int *h, float **pixels)
{
    unsigned char	dummyByte;
    int			i;
    float		max;
    char		token;
    int			width, height;
    float		*rgbPixels;

    fscanf(file, " ");

    skipComments(file);

    if(fscanf(file, "P%c ", &token) != 1) {
         fprintf(stderr, "pnmRead: Had trouble reading PNM tag\n");
	 return 0;
    }

    skipComments(file);

    if(fscanf(file, "%d ", &width) != 1) {
         fprintf(stderr, "pnmRead: Had trouble reading PNM width\n");
	 return 0;
    }

    skipComments(file);

    if(fscanf(file, "%d ", &height) != 1) {
         fprintf(stderr, "pnmRead: Had trouble reading PNM height\n");
	 return 0;
    }

    skipComments(file);

    if(token != '1' && token != '4') {
        if(fscanf(file, "%f", &max) != 1) {
             fprintf(stderr, "pnmRead: Had trouble reading PNM max value\n");
	     return 0;
        }
    }

    rgbPixels = static_cast<float*>(malloc(width * height * 4 * sizeof(float)));
    if(rgbPixels == NULL) {
         fprintf(stderr, "pnmRead: Couldn't allocate %zd bytes\n", width * height * 4 * sizeof(float));
         fprintf(stderr, "pnmRead: (For a %d by %d image)\n", width, height);
	 return 0;
    }

    if(token != '4') {
	skipComments(file);
    }

    if(token != '4') { 
        // ??
        fread(&dummyByte, 1, 1, file);	/* chuck white space */
    }

    if(token == '1')
    {
	for(i = 0; i < width * height; i++)
	{
	    int pixel;
	    fscanf(file, "%d", &pixel);
	    pixel = 1 - pixel;
	    rgbPixels[i * 4 + 0] = pixel;
	    rgbPixels[i * 4 + 1] = pixel;
	    rgbPixels[i * 4 + 2] = pixel;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '2')
    {
	for(i = 0; i < width * height; i++)
	{
	    int pixel;
	    fscanf(file, "%d", &pixel);
	    rgbPixels[i * 4 + 0] = pixel / max;
	    rgbPixels[i * 4 + 1] = pixel / max;
	    rgbPixels[i * 4 + 2] = pixel / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '3')
    {
	for(i = 0; i < width * height; i++)
	{
	    int r, g, b;
	    fscanf(file, "%d %d %d", &r, &g, &b);
	    rgbPixels[i * 4 + 0] = r / max;
	    rgbPixels[i * 4 + 1] = g / max;
	    rgbPixels[i * 4 + 2] = b / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '4')
    {
        int bitnum = 0;

	for(i = 0; i < width * height; i++)
	{
	    unsigned char pixel;
	    unsigned char value = 0;

	    if(bitnum == 0) {
	        fread(&value, 1, 1, file);
            }

	    pixel = (1 - ((value >> (7 - bitnum)) & 1));
	    rgbPixels[i * 4 + 0] = pixel;
	    rgbPixels[i * 4 + 1] = pixel;
	    rgbPixels[i * 4 + 2] = pixel;
	    rgbPixels[i * 4 + 3] = 1.0;

	    if(++bitnum == 8 || ((i + 1) % width) == 0) {
	        bitnum = 0;
            }
	}
    }
    else if(token == '5')
    {
	for(i = 0; i < width * height; i++)
	{
	    unsigned char pixel;
	    fread(&pixel, 1, 1, file);
	    rgbPixels[i * 4 + 0] = pixel / max;
	    rgbPixels[i * 4 + 1] = pixel / max;
	    rgbPixels[i * 4 + 2] = pixel / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '6')
    {
	for(i = 0; i < width * height; i++)
	{
	    unsigned char rgb[3];
	    fread(rgb, 3, 1, file);
	    rgbPixels[i * 4 + 0] = rgb[0] / max;
	    rgbPixels[i * 4 + 1] = rgb[1] / max;
	    rgbPixels[i * 4 + 2] = rgb[2] / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    *w = width;
    *h = height;
    *pixels = rgbPixels;
    return 1;
}

void usage(const char *progName) 
{
    fprintf(stderr, "usage: %s\n", progName);
}

void MakeStubDrawable()
{
    using namespace VulkanApp;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    float specular_color[4] { 0.8f, 0.8f, 0.8f, 0.8f };
    float shininess = 10.0f;

    float white[4] {1, 1, 1, 1};
    float no_texture[2] {0, 0};
    float ll[3] {-1, -1, 0};
    float lr[3] {1, -1, 0};
    float ul[3] {-1, -1, 0};
    float ur[3] {1, -1, 0};
    float n[3] {0, 0, 1};
    vertices.push_back(Vertex(ll, n, white, no_texture));
    vertices.push_back(Vertex(lr, n, white, no_texture));
    vertices.push_back(Vertex(ul, n, white, no_texture));
    vertices.push_back(Vertex(ur, n, white, no_texture));
    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);
    indices.push_back(0);
    indices.push_back(2);
    indices.push_back(3);

    int width = 1, height = 1;
    std::vector<uint8_t> rgba8_unorm = {255, 255, 255, 255};
    auto texture = std::make_shared<RGBA8UNormImage>(width, height, rgba8_unorm);

    drawable = std::make_unique<Drawable>(vertices, indices, specular_color, shininess, texture);

    object_manip = manipulator(drawable->bounds, fov / 180.0f * 3.14159f / 2);
    light_manip = manipulator(aabox(), fov / 180.0f * 3.14159f / 2);

    vec3 boxmin {0, 0, 0};
    vec3 boxmax {1,1,1}; // {512, 512, 114};
    volume_bounds = aabox(boxmin, boxmax);
    volume_manip = manipulator(volume_bounds, fov / 180.0f * 3.14159f / 2);

    if(drawing_mode == DRAW_VULKAN) {
        current_manip = &object_manip;
    } else {
        current_manip = &volume_manip;
    }

}

int main(int argc, char **argv)
{
    using namespace VulkanApp;
    
    MakeStubDrawable();
    LoadCTData();

    be_verbose = (getenv("BE_NOISY") != nullptr);
    enable_validation = (getenv("VALIDATE") != nullptr);

    [[maybe_unused]] const char *progName = argv[0];
    argv++;
    argc--;
    while(argc > 0 && argv[0][0] == '-') {
    }
    // if(argc != 1) {
        // fprintf(stderr, "expected a filename\n");
        // usage(progName);
        // exit(EXIT_FAILURE);
    // }
    // const char *input_filename = argv[0];

    glfwSetErrorCallback(ErrorCallback);

    if(!glfwInit()) {
	std::cerr << "GLFW initialization failed.\n";
        exit(EXIT_FAILURE);
    }

    if (!glfwVulkanSupported()) {
	std::cerr << "GLFW reports Vulkan is not supported\n";
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "vulkan test", nullptr, nullptr);

    VulkanApp::InitializeInstance();

    VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if (err) {
	std::cerr << "GLFW window creation failed " << err << "\n";
        exit(EXIT_FAILURE);
    }

    VulkanApp::InitializeState();

    glfwSetKeyCallback(window, KeyCallback);
    glfwSetMouseButtonCallback(window, ButtonCallback);
    glfwSetCursorPosCallback(window, MotionCallback);
    glfwSetScrollCallback(window, ScrollCallback);
    // glfwSetFramebufferSizeCallback(window, ResizeCallback);
    glfwSetWindowRefreshCallback(window, DrawFrame);

    while (!glfwWindowShouldClose(window)) {

        DrawFrame(window);

        // if(gStreamFrames)
            // glfwPollEvents();
        // else
        glfwWaitEvents();
    }

    Cleanup();

    glfwTerminate();
}

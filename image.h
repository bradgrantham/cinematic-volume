#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <memory>

template <typename T>
VkFormat GetVulkanFormat();

template <>
VkFormat GetVulkanFormat<float>()
{
    return VK_FORMAT_R32_SFLOAT;
}

template <>
VkFormat GetVulkanFormat<int16_t>()
{
    return VK_FORMAT_R16_UNORM;
}

template <>
VkFormat GetVulkanFormat<vec3>()
{
    return VK_FORMAT_R32G32B32_SFLOAT;
}

struct RGBA8UNorm {
    uint8_t r, g, b, a;
    RGBA8UNorm() {}
    RGBA8UNorm(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        r(r), g(g), b(b), a(a)
    {}
    RGBA8UNorm& operator=(const uint8_t color[4])
    {
        r = color[0];
        g = color[1];
        b = color[2];
        a = color[3];
        return *this;
    }
};

template <>
VkFormat GetVulkanFormat<RGBA8UNorm>()
{
    return VK_FORMAT_R8G8B8A8_UNORM;
}

template <class T>
struct Image
{
private:
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    std::vector<T> pixels;
    T clamp_value;

    static constexpr int tile_size = 8;
    uint32_t width_tiles;
    uint32_t height_tiles;
    uint32_t depth_tiles;
    typedef std::array<T, tile_size * tile_size * tile_size> tile;
    std::vector<tile> tiled;

    void MakeTiled()
    {
        width_tiles = (width + tile_size - 1) / tile_size;
        height_tiles = (height + tile_size - 1) / tile_size;
        depth_tiles = (depth + tile_size - 1) / tile_size;
        tiled.resize(width_tiles * height_tiles * depth_tiles);

        for(uint32_t i = 0; i < width; i++) {
            for(uint32_t j = 0; j < height; j++) {
                for(uint32_t k = 0; k < depth; k++) {

                    uint32_t tile_i = i / tile_size;
                    uint32_t tile_j = j / tile_size;
                    uint32_t tile_k = k / tile_size;
                    tile& t = tiled[tile_i + tile_j * width_tiles + tile_k * width_tiles * height_tiles];

                    uint32_t i_ = i % tile_size;
                    uint32_t j_ = j % tile_size;
                    uint32_t k_ = k % tile_size;
                    uint32_t within_tile = i_ + j_ * tile_size + k_ * tile_size * tile_size;

                    uint32_t within_image = i + j * width + k * width * height;

                    t[within_tile] = pixels[within_image];
                }
            }
        }
    }

public:
    Image(int width, int height, int depth, std::vector<T>& pixels, const T& clamp_value) :
        width(width),
        height(height),
        depth(depth),
        pixels(std::move(pixels)),
        clamp_value(clamp_value)
    {
        MakeTiled();
    }

    Image(int width, int height, int depth) :
        width(width),
        height(height),
        depth(depth),
        pixels(width * height * depth)
    {}

    // VkFormat GetVulkanFormat() { return ::GetVulkanFormat(pixels[0]); }
    static VkFormat GetVulkanFormat() { return ::GetVulkanFormat<T>(); }
    int GetWidth() { return width; }
    int GetHeight() { return height; }
    int GetDepth() { return depth; }
    void* GetData() { return pixels.data(); }
    size_t GetSize() { return pixels.size() * sizeof(T); }
    T Sample(const vec3& str);
    void SetPixel(int i, int j, int k, const T& v) {
        pixels[i + j * GetWidth() + k * GetWidth() * GetHeight()] = v;
    }
    T FetchUnchecked(uint32_t i, uint32_t j, uint32_t k);
};

template <class T>
T Image<T>::FetchUnchecked(uint32_t i, uint32_t j, uint32_t k)
{
    uint32_t tile_i = i / tile_size;
    uint32_t tile_j = j / tile_size;
    uint32_t tile_k = k / tile_size;
    tile& t = tiled[tile_i + tile_j * width_tiles + tile_k * width_tiles * height_tiles];

    uint32_t i_ = i % tile_size;
    uint32_t j_ = j % tile_size;
    uint32_t k_ = k % tile_size;
    uint32_t within_tile = i_ + j_ * tile_size + k_ * tile_size * tile_size;

    return t[within_tile];
}

template <class T>
T Image<T>::Sample(const vec3& str)
{
    if((str[0] < 0.0f) || (str[0] >= 1.0f) ||
        (str[1] < 0.0f) || (str[1] >= 1.0f) ||
        (str[2] < 0.0f) || (str[2] >= 1.0f)) {

        return clamp_value;
    }

    float u = str[0] * width;
    uint32_t i0 = std::clamp(static_cast<uint32_t>(u - .5), 0u, width - 1);
    uint32_t i1 = std::clamp(static_cast<uint32_t>(u + .5), 0u, width - 1);
    float a0 = (u <= .5f) ? (1.0f) : ((u >= width - .5f) ? (0.0f) : (1.0f - (u - .5f - i0)));
    float a1 = 1.0f - a0;

    float v = str[1] * height;
    uint32_t j0 = std::clamp(static_cast<uint32_t>(v - .5), 0u, height - 1);
    uint32_t j1 = std::clamp(static_cast<uint32_t>(v + .5), 0u, height - 1);
    float b0 = (v <= .5f) ? (1.0f) : ((v >= height - .5f) ? (0.0f) : (1.0f - (v - .5f - j0)));
    float b1 = 1.0f - b0;

    float w = str[2] * depth;
    uint32_t k0 = std::clamp(static_cast<uint32_t>(w - .5), 0u, depth - 1);
    uint32_t k1 = std::clamp(static_cast<uint32_t>(w + .5), 0u, depth - 1);
    float c0 = (w <= .5f) ? (1.0f) : ((w >= depth - .5f) ? (0.0f) : (1.0f - (w - .5f - k0)));
    float c1 = 1.0f - c0;

    T v000 = FetchUnchecked(i0, j0, k0);
    T v001 = FetchUnchecked(i0, j0, k1);
    T v010 = FetchUnchecked(i0, j1, k0);
    T v011 = FetchUnchecked(i0, j1, k1);
    T v100 = FetchUnchecked(i1, j0, k0);
    T v101 = FetchUnchecked(i1, j0, k1);
    T v110 = FetchUnchecked(i1, j1, k0);
    T v111 = FetchUnchecked(i1, j1, k1);

    T v00 = static_cast<T>(v000 * c0 + v001 * c1);
    T v01 = static_cast<T>(v010 * c0 + v011 * c1);
    T v10 = static_cast<T>(v100 * c0 + v101 * c1);
    T v11 = static_cast<T>(v110 * c0 + v111 * c1);

    T v0 = static_cast<T>(v00 * b0 + v01 * b1);
    T v1 = static_cast<T>(v10 * b0 + v11 * b1);

    T val = static_cast<T>(v0 * a0 + v1 * a1);

    return val;
}

typedef Image<RGBA8UNorm> RGBA8UNormImage;
typedef std::shared_ptr<Image<RGBA8UNorm>> RGBA8UNormImagePtr;

#endif /* __IMAGE_H__ */

#include "camera.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <unistd.h>

#include <linux/videodev2.h>

namespace {

static inline void
xioctl(int fd, unsigned long req, void* arg)
{
  for (;;) {
    if (::ioctl(fd, req, arg) == 0)
      return;
    if (errno == EINTR)
      continue;
    throw std::runtime_error(std::string("v4l2 ioctl failed: ") + std::strerror(errno));
  }
}

[[nodiscard]] static inline auto
clamp_u8(int v)
{
  if (v < 0)
    return 0;
  if (v > 255)
    return 255;
  return v;
}

static inline void
yuv_to_rgb_pixel(int y, int u, int v, unsigned char& r, unsigned char& g, unsigned char& b)
{
  int c = y - 16;
  int d = u - 128;
  int e = v - 128;

  if (c < 0)
    c = 0;

  int rr = (298 * c + 409 * e + 128) >> 8;
  int gg = (298 * c - 100 * d - 208 * e + 128) >> 8;
  int bb = (298 * c + 516 * d + 128) >> 8;

  r = static_cast<unsigned char>(clamp_u8(rr));
  g = static_cast<unsigned char>(clamp_u8(gg));
  b = static_cast<unsigned char>(clamp_u8(bb));
}

struct mapped_buffer final
{
  void* ptr = nullptr;
  size_t len = 0;
};

class v4l2_camera final : public camera
{
public:
  v4l2_camera(const int device_index, const int w, const int h)
    : w_(w)
    , h_(h)
  {
    if ((w_ <= 0) || (h_ <= 0)) {
      throw std::runtime_error("invalid camera dimensions");
    }

    dev_ = "/dev/video" + std::to_string(device_index);

    fd_ = ::open(dev_.c_str(), O_RDWR | O_CLOEXEC);

    if (fd_ < 0) {
      throw std::runtime_error(std::string("open failed: ") + dev_ + ": " + std::strerror(errno));
    }

    v4l2_capability cap{};

    xioctl(fd_, VIDIOC_QUERYCAP, &cap);

    if ((cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0) {
      throw std::runtime_error("device is not a video capture device");
    }

    if ((cap.capabilities & V4L2_CAP_STREAMING) == 0) {
      throw std::runtime_error("device does not support streaming");
    }

    v4l2_format fmt{};

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = static_cast<__u32>(w_);
    fmt.fmt.pix.height = static_cast<__u32>(h_);
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    xioctl(fd_, VIDIOC_S_FMT, &fmt);

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
      throw std::runtime_error("device did not accept YUYV format");
    }

    w_ = static_cast<int>(fmt.fmt.pix.width);
    h_ = static_cast<int>(fmt.fmt.pix.height);

    const size_t yuyv_bytes = static_cast<size_t>(w_) * static_cast<size_t>(h_) * 2ull;

    yuyv_.resize(yuyv_bytes);

    rgb_.resize(static_cast<size_t>(w_) * static_cast<size_t>(h_) * 3ull);

    v4l2_requestbuffers req{};

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    xioctl(fd_, VIDIOC_REQBUFS, &req);

    if (req.count < 2) {
      throw std::runtime_error("insufficient v4l2 buffers");
    }

    bufs_.resize(req.count);

    for (uint32_t i = 0; i < req.count; ++i) {

      v4l2_buffer buf{};

      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;

      xioctl(fd_, VIDIOC_QUERYBUF, &buf);

      bufs_[i].len = buf.length;
      bufs_[i].ptr = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

      if (bufs_[i].ptr == MAP_FAILED) {
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
      }

      xioctl(fd_, VIDIOC_QBUF, &buf);
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    xioctl(fd_, VIDIOC_STREAMON, &type);

    streaming_ = true;
  }

  ~v4l2_camera() override
  {
    try {
      if (fd_ >= 0 && streaming_) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ::ioctl(fd_, VIDIOC_STREAMOFF, &type);
      }
    } catch (...) {
    }

    for (auto& b : bufs_) {
      if (b.ptr && b.ptr != MAP_FAILED)
        ::munmap(b.ptr, b.len);
      b.ptr = nullptr;
      b.len = 0;
    }

    if (fd_ >= 0)
      ::close(fd_);
    fd_ = -1;
  }

  [[nodiscard]] auto get_frame_size() const -> std::pair<int, int> override { return std::make_pair(w_, h_); }

  [[nodiscard]] auto wait_frame() -> unsigned char* override
  {
    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;

    for (;;) {
      int r = ::poll(&pfd, 1, -1);
      if (r < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error(std::string("poll failed: ") + std::strerror(errno));
      }
      if (r == 0)
        continue;
      if ((pfd.revents & POLLIN) == 0)
        continue;

      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      xioctl(fd_, VIDIOC_DQBUF, &buf);

      const size_t n = std::min(static_cast<size_t>(buf.bytesused), yuyv_.size());
      std::memcpy(yuyv_.data(), bufs_[buf.index].ptr, n);
      last_bytes_ = n;

      xioctl(fd_, VIDIOC_QBUF, &buf);

      return to_rgb();
    }
  }

  [[nodiscard]] auto to_rgb() -> unsigned char*
  {
    if (yuyv_.empty()) {
      return nullptr;
    }

    if (last_bytes_ < static_cast<size_t>(w_) * static_cast<size_t>(h_) * 2ull) {
      return nullptr;
    }

    const int w = w_;
    const int h = h_;
    const unsigned char* src = yuyv_.data();
    unsigned char* dst = rgb_.data();

    const int pairs_per_row = w / 2;
    const int row_stride_src = w * 2;
    const int row_stride_dst = w * 3;

#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
      const unsigned char* srow = src + static_cast<size_t>(y) * static_cast<size_t>(row_stride_src);
      unsigned char* drow = dst + static_cast<size_t>(y) * static_cast<size_t>(row_stride_dst);

      for (int p = 0; p < pairs_per_row; ++p) {
        const int i = p * 4;
        int y0 = srow[i + 0];
        int u = srow[i + 1];
        int y1 = srow[i + 2];
        int v = srow[i + 3];

        unsigned char r0, g0, b0, r1, g1, b1;
        yuv_to_rgb_pixel(y0, u, v, r0, g0, b0);
        yuv_to_rgb_pixel(y1, u, v, r1, g1, b1);

        const int j = p * 6;
        drow[j + 0] = r0;
        drow[j + 1] = g0;
        drow[j + 2] = b0;
        drow[j + 3] = r1;
        drow[j + 4] = g1;
        drow[j + 5] = b1;
      }

      if ((w & 1) != 0) {
        const int last = (w - 1);
        const int si = last * 2;
        int yy = srow[si + 0];
        int u = srow[si + 1];
        int v = (si + 3 < row_stride_src) ? srow[si + 3] : srow[si + 1];
        unsigned char r, g, b;
        yuv_to_rgb_pixel(yy, u, v, r, g, b);
        const int dj = last * 3;
        drow[dj + 0] = r;
        drow[dj + 1] = g;
        drow[dj + 2] = b;
      }
    }

    return rgb_.data();
  }

private:
  std::string dev_;

  int fd_ = -1;

  int w_ = 0;

  int h_ = 0;

  bool streaming_ = false;

  std::vector<mapped_buffer> bufs_;

  std::vector<unsigned char> yuyv_;

  std::vector<unsigned char> rgb_;

  size_t last_bytes_ = 0;
};

} // namespace

auto
camera::create(const int device_index, const int w, const int h) -> std::unique_ptr<camera>
{
  return std::make_unique<v4l2_camera>(device_index, w, h);
}

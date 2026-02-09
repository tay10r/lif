#include "camera.h"

#include <spdlog/spdlog.h>

#include <linket_config.h>
#include <linket_encode.h>

#include <uv.h>

#include <chrono>
#include <stdexcept>
#include <vector>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "stream_options.h"

namespace {

class program;

struct tile_context final
{
  program* parent{};

  /**
   * @brief When this buffer is non-empty, the tile is being used and not available.
   * */
  uv_buf_t buffer{};

  uv_udp_send_t writer{};

  char data[1400]{};

  [[nodiscard]] auto available() const -> bool { return buffer.len == 0; }
};

class program final
{
public:
  explicit program(const int device_index, const int w, const int h, const char* send_ip, const int send_port)
    : camera_(camera::create(device_index, w, h))
  {
    const auto [w1, h1] = camera_->get_frame_size();

    frame_width_ = w1;

    frame_height_ = h1;

    uv_loop_init(&loop_);

    uv_udp_init(&loop_, &socket_);

    uv_ip4_addr(send_ip, send_port, &send_address_);

    required_tiles_ = (w / LINKET_TILE_SIZE) * (h / LINKET_TILE_SIZE);

    available_tiles_ = required_tiles_ * 2;

    tiles_.resize(available_tiles_);

    for (size_t i = 0; i < tiles_.size(); i++) {
      tiles_[i].parent = this;
    }
  }

  void run()
  {
    while (true) {

      uv_run(&loop_, UV_RUN_NOWAIT);

      auto* rgb = camera_->wait_frame();

      if (!can_send_frame()) {
        SPDLOG_WARN("not enough tiles available for sending frame");
        continue;
      }

      last_frame_time_ = unix_time_us();

      linket_encode(rgb, frame_width_, frame_height_, this, on_encoded_tile);

      SPDLOG_DEBUG("sent frame");
    }

    uv_close(reinterpret_cast<uv_handle_t*>(&socket_), nullptr);

    uv_run(&loop_, UV_RUN_DEFAULT);
  }

protected:
  [[nodiscard]] auto can_send_frame() const -> bool { return available_tiles_ >= required_tiles_; }

  static void on_send(uv_udp_send_t* writer, const int status)
  {
    auto* ctx = static_cast<tile_context*>(uv_handle_get_data(reinterpret_cast<uv_handle_t*>(writer)));
    ctx->buffer.base = nullptr;
    ctx->buffer.len = 0;
    ctx->parent->available_tiles_++;
  }

  static void on_encoded_tile(void* self_ptr, const int x, const int y, const unsigned char* latent)
  {
    auto* self = static_cast<program*>(self_ptr);

    auto* ctx = self->find_tile_context();
    if (!ctx) {
      SPDLOG_ERROR("failed to find tile for encoding");
      return;
    }

    auto* ptr = reinterpret_cast<unsigned char*>(ctx->data);

    *reinterpret_cast<int64_t*>(ptr) = self->last_frame_time_;
    *reinterpret_cast<int32_t*>(ptr + 8) = x;
    *reinterpret_cast<int32_t*>(ptr + 12) = y;

    static_assert((LINKET_BYTES_PER_TILE + 16) <= sizeof(tile_context::data));

    memcpy(ptr + 16, latent, LINKET_BYTES_PER_TILE);

    ctx->buffer.base = reinterpret_cast<char*>(ptr);

    ctx->buffer.len = LINKET_BYTES_PER_TILE + 16;

    uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&ctx->writer), ctx);

    const int err = uv_udp_send(
      &ctx->writer, &self->socket_, &ctx->buffer, 1, reinterpret_cast<const sockaddr*>(&self->send_address_), on_send);
    if (err != 0) {
      SPDLOG_ERROR("failed to send tile");
      ctx->buffer.len = 0;
      return;
    }

    self->available_tiles_--;
  }

  [[nodiscard]] auto find_tile_context() -> tile_context*
  {
    for (size_t i = 0; i < tiles_.size(); i++) {
      auto* ctx = &tiles_[i];
      if (ctx->available()) {
        return ctx;
      }
    }

    return nullptr;
  }

  [[nodiscard]] static auto unix_time_us() -> int64_t
  {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  }

private:
  std::unique_ptr<camera> camera_;

  uv_loop_t loop_;

  uv_udp_t socket_;

  int frame_width_{};

  int frame_height_{};

  int64_t last_frame_time_{};

  std::vector<tile_context> tiles_;

  sockaddr_in send_address_;

  size_t required_tiles_{};

  size_t available_tiles_{};
};

} // namespace

auto
main(const int argc, char** argv) -> int
{
  stream_options opts;

  try {
    if (!opts.parse(argc, argv)) {
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR(e.what());
    return EXIT_FAILURE;
  }

  program prg(opts.video_device, opts.frame_width, opts.frame_height, opts.send_ip.c_str(), opts.send_port);

  SPDLOG_INFO("publishing tiles over UDP to {}:{}", opts.send_ip, opts.send_port);

  prg.run();

  return EXIT_SUCCESS;
}

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

struct tile_buffer final
{
  char data[1408]{};
};

class program final
{
public:
  explicit program(const int device_index,
                   const int w,
                   const int h,
                   const char* send_ip,
                   const int send_port,
                   const bool benchmark)
    : camera_(camera::create(device_index, w, h))
    , benchmark_(benchmark)
  {
    const auto [w1, h1] = camera_->get_frame_size();

    frame_width_ = w1;

    frame_height_ = h1;

    uv_loop_init(&loop_);

    uv_udp_init(&loop_, &socket_);

    sockaddr_in bind_address{};

    uv_ip4_addr("0.0.0.0", 0, &bind_address);

    uv_udp_bind(&socket_, reinterpret_cast<const sockaddr*>(&bind_address), 0);

    uv_ip4_addr(send_ip, send_port, &send_address_);

    const auto num_tiles = (w / LINKET_TILE_SIZE) * (h / LINKET_TILE_SIZE);

    tiles_.resize(num_tiles);

    buffers_.resize(num_tiles);
  }

  void run()
  {
    using clock = std::chrono::high_resolution_clock;

    auto t0 = clock::now();

    while (true) {

      uv_run(&loop_, UV_RUN_NOWAIT);

      auto* rgb = camera_->wait_frame();

      last_frame_time_ = unix_time_us();

      tile_offset_ = 0;

      linket_encode(rgb, frame_width_, frame_height_, this, on_encoded_tile);

      const auto t1 = clock::now();

      uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&writer_), this);

      std::vector<unsigned int> counts(buffers_.size());
      std::vector<sockaddr*> addresses(buffers_.size());
      std::vector<uv_buf_t*> buffers(buffers_.size());
      for (size_t i = 0; i < buffers_.size(); i++) {
        counts[i] = 1;
        addresses[i] = reinterpret_cast<sockaddr*>(&send_address_);
        buffers[i] = &buffers_[i];
      }

      const int num_sent = uv_udp_try_send2(
        &socket_, static_cast<unsigned int>(buffers_.size()), buffers.data(), counts.data(), addresses.data(), 0);
      if (num_sent < 0) {
        SPDLOG_ERROR("failed to send tile {}", uv_strerror(num_sent));
        return;
      } else if (num_sent != static_cast<int>(buffers_.size())) {
        SPDLOG_WARN("only sent {} out of {} tiles", num_sent, buffers_.size());
      }

      const auto t2 = clock::now();

      const auto compression_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      const auto frame_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();

      t0 = t2;

      if (benchmark_) {
        SPDLOG_INFO("total frame time: {:4} ms, compression time: {:4} ms", frame_dt, compression_dt);
      }
    }

    uv_close(reinterpret_cast<uv_handle_t*>(&socket_), nullptr);

    uv_run(&loop_, UV_RUN_DEFAULT);
  }

protected:
  static void on_send(uv_udp_send_t* writer, const int status)
  {
    auto* self = static_cast<program*>(uv_handle_get_data(reinterpret_cast<uv_handle_t*>(writer)));

    (void)self;

    if (status != 0) {
      SPDLOG_ERROR("failed to send tile data: {}\n", uv_strerror(status));
    }
  }

  static void on_encoded_tile(void* self_ptr, const int x, const int y, const unsigned char* latent)
  {
    auto* self = static_cast<program*>(self_ptr);

    auto* tile_buf = &self->tiles_[self->tile_offset_];

    auto* ptr = reinterpret_cast<unsigned char*>(tile_buf->data);

    *reinterpret_cast<int64_t*>(ptr) = self->last_frame_time_;
    *reinterpret_cast<int32_t*>(ptr + 8) = x;
    *reinterpret_cast<int32_t*>(ptr + 12) = y;

    static_assert((LINKET_BYTES_PER_TILE + 16) <= sizeof(tile_buffer::data));

    memcpy(ptr + 16, latent, LINKET_BYTES_PER_TILE);

    self->buffers_[self->tile_offset_].base = reinterpret_cast<char*>(ptr);

    self->buffers_[self->tile_offset_].len = LINKET_BYTES_PER_TILE + 16;

    self->tile_offset_++;
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

  size_t tile_offset_{};

  std::vector<tile_buffer> tiles_;

  std::vector<uv_buf_t> buffers_;

  uv_udp_send_t writer_;

  sockaddr_in send_address_;

  bool benchmark_{ false };
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

  program prg(
    opts.video_device, opts.frame_width, opts.frame_height, opts.send_ip.c_str(), opts.send_port, opts.benchmark);

  SPDLOG_INFO("publishing tiles over UDP to {}:{}", opts.send_ip, opts.send_port);

  prg.run();

  return EXIT_SUCCESS;
}

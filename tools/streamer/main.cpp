#include <spdlog/spdlog.h>

#include <NICE.h>

#include "camera.h"
#include "options.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include <uv.h>

#include <stdlib.h>
#include <string.h>

namespace {

struct frame final
{
  std::unique_ptr<uint8_t[]> rgb;

  int frame_width{};

  int frame_height{};
};

class output_worker final
{
public:
  output_worker(const options& opts)
    : options_(opts)
    , thread_(&output_worker::run_thread, this)
  {
    //
  }

  [[nodiscard]] auto can_queue_frame() const -> bool { return frame_queue_ready_.load(); }

  void queue_frame(std::unique_ptr<frame> fr)
  {
    {
      std::lock_guard<std::mutex> lock(frame_lock_);
      frame_ = std::move(fr);
    }

    uv_async_send(&frame_signal_);
  }

protected:
  static void on_frame(uv_async_t* handle)
  {
    auto* self = static_cast<output_worker*>(uv_handle_get_data(reinterpret_cast<uv_handle_t*>(handle)));

    self->frame_queue_ready_.store(false);

    std::unique_ptr<frame> fr;

    {
      std::lock_guard<std::mutex> lock(self->frame_lock_);
      fr = std::move(self->frame_);
    }

    if (!fr) {
      self->frame_queue_ready_.store(true);
      return;
    }

    self->compress_frame(*fr);

    self->begin_sending_frame();
  }

  void begin_sending_frame()
  {
    num_sent_ = 0;

    send_tile();
  }

  void compress_frame(const frame& fr)
  {
#define ROUND_TILE_SIZE(x) ((((x) + (NICE_TILE_SIZE - 1)) / NICE_TILE_SIZE) * NICE_TILE_SIZE)

    const auto pad_w = ROUND_TILE_SIZE(fr.frame_width);
    const auto pad_h = ROUND_TILE_SIZE(fr.frame_height);

    padded_.resize(pad_w * pad_h * 3, 0);

    for (int y = 0; y < fr.frame_height; y++) {
      memcpy(padded_.data() + y * pad_w * 3, fr.rgb.get() + y * fr.frame_width * 3, fr.frame_width * 3);
    }

    const int num_tiles = (pad_h / NICE_TILE_SIZE) * (pad_w / NICE_TILE_SIZE);

    send_buffers_.resize(num_tiles);

    size_t i = 0;

    for (int y = 0; y < pad_h; y += NICE_TILE_SIZE) {

      for (int x = 0; x < pad_w; x += NICE_TILE_SIZE) {

        auto* buffer = send_buffers_[i].data();

        *reinterpret_cast<uint64_t*>(buffer) = uv_now(&loop_);
        *reinterpret_cast<uint32_t*>(&buffer[8]) = x;
        *reinterpret_cast<uint32_t*>(&buffer[12]) = y;

        unsigned char bits[100 * 12];

        NICE_EncodeTile(engine_, padded_.data() + (y * pad_w + x) * 3, pad_w * 3, &buffer[16]);

        i++;
      }
    }
  }

  void on_failure() { frame_queue_ready_.store(true); }

  static void on_send(uv_udp_send_t* writer, const int status)
  {
    auto* self = static_cast<output_worker*>(uv_handle_get_data(reinterpret_cast<uv_handle_t*>(writer)));

    if (status != 0) {
      SPDLOG_ERROR("failed to send UDP packet: {}", uv_strerror(status));
      self->on_failure();
      return;
    }

    self->num_sent_++;

    if (self->num_sent_ < self->send_buffers_.size()) {
      self->send_tile();
    } else if (self->num_sent_ == self->send_buffers_.size()) {
      self->frame_queue_ready_.store(true);
    }
  }

  void send_tile()
  {
    send_buffer_.base = reinterpret_cast<char*>(send_buffers_.at(num_sent_).data());
    send_buffer_.len = 1216;

    const auto err =
      uv_udp_send(&writer_, &socket_, &send_buffer_, 1, reinterpret_cast<const sockaddr*>(&send_address_), on_send);
    if (err != 0) {
      SPDLOG_ERROR("failed to start sending UDP packet: {}\n", uv_strerror(err));
      on_failure();
      return;
    }
  }

  void run_thread()
  {
    uv_loop_init(&loop_);

    uv_async_init(&loop_, &frame_signal_, on_frame);

    uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&frame_signal_), this);

    uv_udp_init(&loop_, &socket_);

    uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&socket_), this);

    uv_ip4_addr(options_.send_ip.c_str(), options_.send_port, &send_address_);

    uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&writer_), this);

    frame_queue_ready_.store(true);

    uv_run(&loop_, UV_RUN_DEFAULT);

    uv_close(reinterpret_cast<uv_handle_t*>(&frame_signal_), nullptr);

    uv_close(reinterpret_cast<uv_handle_t*>(&socket_), nullptr);

    uv_run(&loop_, UV_RUN_DEFAULT);

    uv_loop_close(&loop_);
  }

private:
  const options options_;

  std::thread thread_;

  uv_loop_t loop_{};

  uv_async_t frame_signal_{};

  uv_udp_t socket_{};

  uv_udp_send_t writer_{};

  uv_buf_t send_buffer_{};

  sockaddr_in send_address_{};

  std::vector<std::array<unsigned char, 1400>> send_buffers_;

  size_t num_sent_{};

  std::mutex frame_lock_;

  std::unique_ptr<frame> frame_;

  std::atomic<bool> frame_queue_ready_{ false };

  std::vector<uint8_t> padded_;

  NICE_Engine* engine_{ NICE_NewEngine() };
};

class program final
{
public:
  program(const options& opts)
    : worker_(opts)
    , camera_(camera::create(opts.video_device, opts.frame_width, opts.frame_height))
  {
    //
  }

  void run()
  {
    const auto frame_size = camera_->get_frame_size();

    while (true) {

      auto* rgb_ptr = camera_->wait_frame();
      if (!rgb_ptr) {
        SPDLOG_ERROR("failed to grab frame");
        return;
      }

      if (worker_.can_queue_frame()) {

        auto fr = std::make_unique<frame>();
        fr->rgb.reset(new uint8_t[frame_size.first * frame_size.second * 3]);
        fr->frame_width = frame_size.first;
        fr->frame_height = frame_size.second;
        memcpy(fr->rgb.get(), rgb_ptr, frame_size.first * frame_size.second * 3);

        worker_.queue_frame(std::move(fr));
      }
    }
  }

private:
  output_worker worker_;

  std::unique_ptr<camera> camera_;
};

} // namespace

auto
main(int argc, char** argv) -> int
{
  options opts;

  if (!opts.parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  program prg(opts);

  prg.run();

  return EXIT_SUCCESS;
}

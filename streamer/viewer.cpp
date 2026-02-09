#include <glow/main.hpp>

#include <linket_config.h>
#include <linket_decode.h>

#include <imgui_stdlib.h>
#include <implot.h>

#include <glad/glad.h>

#include <uv.h>

#include <stdint.h>

#include <mutex>
#include <thread>
#include <vector>

namespace {

struct connection_config final
{
  std::string ip{ "127.0.0.1" };

  int port{ 9850 };

  int width{ 1280 };

  int height{ 720 };
};

struct io_plan final
{
  std::vector<connection_config> connections;
};

struct tile final
{
  int x{};

  int y{};

  uint8_t rgb[LINKET_TILE_SIZE * LINKET_TILE_SIZE * 3];
};

struct connection final
{
  uv_udp_t socket{};

  uint8_t recv_buffer[2048];

  std::mutex tile_lock;

  std::vector<tile> tile_queue;

  int64_t timestamp{};

  void handle_new_timestamp()
  {
    {
      std::lock_guard<std::mutex> lock(tile_lock);
      tile_queue.clear();
    }
  }
};

class io_worker final
{
public:
  io_worker(const io_plan& plan)
    : plan_(plan)
    , connections_(plan.connections.size())
    , thread_(&io_worker::run_thread, this)
  {
  }

  void stop()
  {
    uv_async_send(&stop_signal_);

    if (thread_.joinable()) {
      thread_.join();
    }
  }

  [[nodiscard]] auto get_tile_queue(const size_t index) -> std::vector<tile>
  {
    auto& c = connections_.at(index);
    if (!c) {
      return {};
    }

    std::vector<tile> t;

    {
      std::lock_guard<std::mutex> m(c->tile_lock);
      t = std::move(c->tile_queue);
    }

    return t;
  }

protected:
  static void on_alloc(uv_handle_t* handle, const size_t, uv_buf_t* buf)
  {
    auto* conn = static_cast<connection*>(uv_handle_get_data(handle));
    buf->base = reinterpret_cast<char*>(conn->recv_buffer);
    buf->len = sizeof(conn->recv_buffer);
  }

  static void on_read(uv_udp_t* socket,
                      const ssize_t read_size,
                      const uv_buf_t*,
                      const sockaddr* sender,
                      const unsigned int flags)
  {
    if (read_size != (LINKET_BYTES_PER_TILE + 16)) {
      return;
    }

    auto* conn = static_cast<connection*>(uv_handle_get_data(reinterpret_cast<const uv_handle_t*>(socket)));

    auto* ptr = &conn->recv_buffer[0];

    const auto timestamp = *reinterpret_cast<const int64_t*>(ptr);
    if (timestamp > conn->timestamp) {
      // new tile
      conn->timestamp = timestamp;
      conn->handle_new_timestamp();
    } else if (timestamp < conn->timestamp) {
      // old tile
      return;
    }

    tile t;

    t.x = *reinterpret_cast<const int32_t*>(ptr + 8);
    t.y = *reinterpret_cast<const int32_t*>(ptr + 12);

    linket_decode_tile(0, 0, ptr + 16, LINKET_TILE_SIZE, LINKET_TILE_SIZE, t.rgb);

    {
      std::lock_guard<std::mutex> lock(conn->tile_lock);
      conn->tile_queue.emplace_back(std::move(t));
    }
  }

  void run_thread()
  {
    uv_loop_init(&loop_);

    uv_async_init(&loop_, &stop_signal_, on_stop_signal);

    for (size_t i = 0; i < plan_.connections.size(); i++) {

      auto& c = plan_.connections[i];

      auto& conn = connections_[i];

      conn.reset(new connection());

      uv_udp_init(&loop_, &conn->socket);

      uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&conn->socket), conn.get());

      sockaddr_in bind_address{};

      uv_ip4_addr(c.ip.c_str(), c.port, &bind_address);

      uv_udp_bind(&conn->socket, reinterpret_cast<const sockaddr*>(&bind_address), 0);

      uv_udp_recv_start(&conn->socket, on_alloc, on_read);
    }

    uv_run(&loop_, UV_RUN_DEFAULT);

    for (auto& c : connections_) {
      uv_close(reinterpret_cast<uv_handle_t*>(&c->socket), nullptr);
    }

    uv_close(reinterpret_cast<uv_handle_t*>(&stop_signal_), nullptr);

    uv_run(&loop_, UV_RUN_DEFAULT);

    uv_loop_close(&loop_);
  }

  static void on_stop_signal(uv_async_t* handle)
  {
    auto* loop = uv_handle_get_loop(reinterpret_cast<uv_handle_t*>(handle));

    uv_stop(loop);
  }

private:
  io_plan plan_;

  std::vector<std::unique_ptr<connection>> connections_;

  std::thread thread_;

  uv_loop_t loop_{};

  uv_async_t stop_signal_{};
};

struct frame final
{
  GLuint texture{};

  int width{};

  int height{};
};

class viewer final
{
public:
  viewer(const io_plan& p)
  {
    frames_.resize(p.connections.size());

    for (size_t i = 0; i < frames_.size(); i++) {

      auto& f = frames_[i];

      f.width = p.connections[i].width;
      f.height = p.connections[i].height;

      glGenTextures(1, &f.texture);

      glBindTexture(GL_TEXTURE_2D, f.texture);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, f.width, f.height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    }
  }

  void teardown()
  {
    for (auto& f : frames_) {
      glDeleteTextures(1, &f.texture);
    }
  }

  void loop(io_worker& io)
  {
    for (size_t i = 0; i < frames_.size(); i++) {
      update_frame(frames_[i], io.get_tile_queue(i));
    }

    for (size_t i = 0; i < frames_.size(); i++) {

      ImGui::PushID(i);

      if (ImGui::Begin("##viewer")) {
        render_frame(frames_[i]);
      }

      ImGui::End();

      ImGui::PopID();
    }
  }

protected:
  void update_frame(frame& f, std::vector<tile> tile_queue)
  {
    glBindTexture(GL_TEXTURE_2D, f.texture);

    for (auto& t : tile_queue) {

      if (((t.x + LINKET_TILE_SIZE) > f.width) || ((t.y + LINKET_TILE_SIZE) > f.height)) {
        continue;
      }

      glTexSubImage2D(GL_TEXTURE_2D, 0, t.x, t.y, LINKET_TILE_SIZE, LINKET_TILE_SIZE, GL_RGB, GL_UNSIGNED_BYTE, t.rgb);
    }
  }

  void render_frame(const frame& f)
  {
    if (!ImPlot::BeginPlot("##plot", ImVec2(-1, -1), ImPlotFlags_NoFrame | ImPlotFlags_Equal)) {
      return;
    }

    GLuint tex = f.texture;

    ImPlot::PlotImage("##image", reinterpret_cast<ImTextureID>(tex), ImPlotPoint(0, 0), ImPlotPoint(f.width, f.height));

    ImPlot::EndPlot();
  }

private:
  std::vector<frame> frames_;
};

class controller_widget final
{
public:
  controller_widget() { io_plan_.connections.emplace_back(); }

  void loop()
  {
    ImGui::SeparatorText("Connections");

    for (size_t i = 0; i < io_plan_.connections.size(); i++) {

      ImGui::PushID(static_cast<int>(i));

      auto& c = io_plan_.connections[i];

      ImGui::InputText("Address", &c.ip);

      ImGui::InputInt("Port", &c.port);

      ImGui::InputInt("Width", &c.width);

      ImGui::InputInt("Height", &c.height);

      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8F, 0, 0, 1));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9F, 0, 0, 1));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0F, 0, 0, 1));
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));

      ImGui::Button("Remove");

      ImGui::PopStyleColor(4);

      ImGui::Separator();

      ImGui::PopID();
    }

    if (ImGui::Button("Add")) {
      io_plan_.connections.emplace_back();
    }

    ImGui::SameLine();

    connect_requested_ = ImGui::Button("Connect");
  }

  [[nodiscard]] auto connect_requested() const -> bool { return connect_requested_; }

  [[nodiscard]] auto plan() const -> const io_plan& { return io_plan_; }

private:
  io_plan io_plan_;

  bool connect_requested_{ false };
};

class app_impl final : public glow::app
{
public:
  void setup(glow::platform&) override {}

  void teardown(glow::platform&) override { stop_io(); }

  void loop(glow::platform&) override
  {
    if (!io_worker_) {
      loop_controller();
    }

    if (viewer_ && io_worker_) {
      viewer_->loop(*io_worker_);
    }
  }

protected:
  void stop_io()
  {
    if (io_worker_) {
      io_worker_->stop();
      io_worker_.reset();
    }

    if (viewer_) {
      viewer_->teardown();
      viewer_.reset();
    }
  }

  void connect()
  {
    io_worker_.reset(new io_worker(controller_.plan()));

    viewer_.reset(new viewer(controller_.plan()));
  }

  void loop_controller()
  {
    if (ImGui::Begin("Controller")) {
      controller_.loop();

      if (controller_.connect_requested()) {
        connect();
      }
    }

    ImGui::End();
  }

private:
  controller_widget controller_;

  std::unique_ptr<io_worker> io_worker_;

  std::unique_ptr<viewer> viewer_;
};

} // namespace

GLOW_APP(app_impl);

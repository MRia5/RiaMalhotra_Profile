#pragma once
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <openssl/ssl.h>

/**
 * WebSocketClient
 * ---------------
 * Persistent TLS WebSocket client for Deribit JSON-RPC API.
 * Handles connect, send, receive loop, and auto-reconnect.
 */
class WebSocketClient {
public:
    using MessageCallback = std::function<void(const std::string&)>;
    using ErrorCallback   = std::function<void(const std::string&)>;

    WebSocketClient(const std::string& host,
                    const std::string& port,
                    const std::string& path);
    ~WebSocketClient();

    // Connect and start receive loop in background thread
    bool connect();

    // Send a raw JSON-RPC string
    bool send(const std::string& message);

    // Disconnect cleanly
    void disconnect();

    // Register callbacks
    void on_message(MessageCallback cb) { message_cb_ = cb; }
    void on_error(ErrorCallback cb)     { error_cb_   = cb; }

    bool is_connected() const { return connected_.load(); }

private:
    std::string host_, port_, path_;
    int sockfd_ = -1;
    SSL_CTX* ssl_ctx_ = nullptr;
    SSL*     ssl_     = nullptr;

    std::atomic<bool> connected_{false};
    std::atomic<bool> stop_{false};
    std::thread recv_thread_;

    MessageCallback message_cb_;
    ErrorCallback   error_cb_;

    bool tls_handshake();
    bool ws_handshake();
    void receive_loop();
    std::string build_ws_frame(const std::string& payload);
    std::string parse_ws_frame(const std::string& raw);
};

#pragma once
#include "websocket_client.h"
#include <string>
#include <map>
#include <functional>
#include <mutex>
#include <atomic>

enum class OrderSide   { BUY, SELL };
enum class OrderType   { MARKET, LIMIT };
enum class OrderStatus { PENDING, OPEN, FILLED, CANCELLED, REJECTED };

struct Order {
    std::string  order_id;
    std::string  instrument;
    OrderSide    side;
    OrderType    type;
    double       price    = 0.0;  // 0 for market orders
    double       amount   = 0.0;
    double       filled   = 0.0;
    OrderStatus  status   = OrderStatus::PENDING;
    long long    timestamp = 0;

    double remaining() const { return amount - filled; }
    bool   is_done()   const {
        return status == OrderStatus::FILLED
            || status == OrderStatus::CANCELLED
            || status == OrderStatus::REJECTED;
    }
};

/**
 * OrderManager
 * ------------
 * Submits, tracks, and cancels orders via Deribit JSON-RPC.
 * All order state is kept in an in-memory map (order_id → Order).
 */
class OrderManager {
public:
    using FillCallback   = std::function<void(const Order&)>;
    using StatusCallback = std::function<void(const Order&)>;

    explicit OrderManager(WebSocketClient& ws);

    // Place a market order; returns order_id or empty string on failure
    std::string market_order(const std::string& instrument,
                             OrderSide side,
                             double amount);

    // Place a limit order
    std::string limit_order(const std::string& instrument,
                            OrderSide side,
                            double amount,
                            double price);

    // Cancel an open order
    bool cancel_order(const std::string& order_id);

    // Cancel all open orders for an instrument
    void cancel_all(const std::string& instrument);

    // Get order snapshot (thread-safe)
    Order get_order(const std::string& order_id) const;

    // Register callbacks
    void on_fill(FillCallback cb)     { fill_cb_   = cb; }
    void on_status(StatusCallback cb) { status_cb_ = cb; }

private:
    WebSocketClient& ws_;

    mutable std::mutex order_mutex_;
    std::map<std::string, Order> orders_;
    std::atomic<int> req_id_{1000};

    FillCallback   fill_cb_;
    StatusCallback status_cb_;

    void on_message(const std::string& raw);
    void handle_order_update(const std::string& json);
    int  next_id() { return req_id_.fetch_add(1); }

    std::string build_buy_rpc(int id, const std::string& instrument,
                               double amount, double price, const std::string& type);
    std::string build_sell_rpc(int id, const std::string& instrument,
                                double amount, double price, const std::string& type);
    std::string build_cancel_rpc(int id, const std::string& order_id);
};

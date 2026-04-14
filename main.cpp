#include "websocket_client.h"
#include "market_data_feed.h"
#include "order_manager.h"
#include "position_tracker.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

// ─────────────────────────────────────────────────────────
//  CONFIG — edit before running
//  Use Deribit TESTNET: test.deribit.com (port 443)
//  Never connect to www.deribit.com without proper risk controls
// ─────────────────────────────────────────────────────────
static const std::string HOST       = "test.deribit.com";
static const std::string PORT       = "443";
static const std::string PATH       = "/ws/api/v2";
static const std::string INSTRUMENT = "BTC-PERPETUAL";   // change as needed
static const double      MAX_QTY    = 10.0;              // position limit (USD contracts)

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    std::cout << "\n[main] Shutting down...\n";
    g_running = false;
}

int main() {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "======================================\n";
    std::cout << "  FirstTrade Crypto Engine (C++)\n";
    std::cout << "  Host : " << HOST << "\n";
    std::cout << "  Inst : " << INSTRUMENT << "\n";
    std::cout << "  ⚠️   TESTNET mode only\n";
    std::cout << "======================================\n\n";

    // 1. WebSocket connection
    WebSocketClient ws(HOST, PORT, PATH);

    if (!ws.connect()) {
        std::cerr << "[main] ❌ WebSocket connection failed.\n";
        return 1;
    }
    std::cout << "[main] ✅ Connected to Deribit\n";

    // 2. Market data feed
    MarketDataFeed feed(ws);
    PositionTracker tracker;
    OrderManager    orders(ws);

    // Wire fill events into position tracker
    orders.on_fill([&tracker](const Order& o) {
        tracker.on_fill(o);
        std::cout << "[fill] " << (o.side == OrderSide::BUY ? "BUY" : "SELL")
                  << " " << o.filled << " @ " << o.price
                  << "  instrument=" << o.instrument << "\n";
    });

    // Subscribe to order book; update mark price on each tick
    feed.subscribe_book(INSTRUMENT, "10", [&](const OrderBook& book) {
        tracker.update_mark(INSTRUMENT, book.mid_price());

        // Risk gate
        if (!tracker.check_limits(MAX_QTY)) {
            std::cerr << "[risk] ⚠️ Position limit exceeded — cancelling all.\n";
            orders.cancel_all(INSTRUMENT);
        }
    });

    feed.subscribe_ticker(INSTRUMENT, [](const Ticker& t) {
        std::cout << "[tick] " << t.instrument
                  << "  last=" << t.last_price
                  << "  mark=" << t.mark_price
                  << "  OI="   << t.open_interest << "\n";
    });

    // 3. Main loop — print position summary every 10 seconds
    while (g_running && ws.is_connected()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));

        std::cout << "\n────── Position Summary ──────\n";
        tracker.print_summary();
        std::cout << "  Total unrealised PnL : "
                  << tracker.total_unrealised_pnl() << " USD\n";
        std::cout << "  Total realised PnL   : "
                  << tracker.total_realised_pnl() << " USD\n";
        std::cout << "──────────────────────────────\n\n";
    }

    ws.disconnect();
    std::cout << "[main] Disconnected. Goodbye.\n";
    return 0;
}

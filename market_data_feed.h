#pragma once
#include "websocket_client.h"
#include <string>
#include <map>
#include <functional>
#include <mutex>

/**
 * OrderBook
 * ---------
 * Level-2 order book: price → size mappings for bids and asks.
 */
struct OrderBook {
    std::map<double, double, std::greater<double>> bids; // price desc
    std::map<double, double>                       asks; // price asc
    std::string instrument;
    long long   timestamp = 0;

    double best_bid()  const { return bids.empty()  ? 0.0 : bids.begin()->first;  }
    double best_ask()  const { return asks.empty()  ? 0.0 : asks.begin()->first;  }
    double mid_price() const { return (best_bid() + best_ask()) / 2.0; }
    double spread()    const { return best_ask() - best_bid(); }
};

/**
 * Ticker
 * ------
 * Real-time snapshot from the ticker channel.
 */
struct Ticker {
    std::string instrument;
    double last_price    = 0.0;
    double mark_price    = 0.0;
    double index_price   = 0.0;
    double volume_24h    = 0.0;
    double open_interest = 0.0;
    long long timestamp  = 0;
};

/**
 * MarketDataFeed
 * --------------
 * Subscribes to Deribit WebSocket channels and maintains live
 * order books and ticker snapshots.
 */
class MarketDataFeed {
public:
    using BookCallback   = std::function<void(const OrderBook&)>;
    using TickerCallback = std::function<void(const Ticker&)>;

    explicit MarketDataFeed(WebSocketClient& ws);

    // Subscribe to order book updates for an instrument
    void subscribe_book(const std::string& instrument,
                        const std::string& depth = "10",
                        BookCallback cb = nullptr);

    // Subscribe to ticker channel
    void subscribe_ticker(const std::string& instrument,
                          TickerCallback cb = nullptr);

    // Access latest snapshots (thread-safe)
    OrderBook get_book(const std::string& instrument) const;
    Ticker    get_ticker(const std::string& instrument) const;

private:
    WebSocketClient& ws_;

    mutable std::mutex book_mutex_;
    mutable std::mutex ticker_mutex_;

    std::map<std::string, OrderBook>   books_;
    std::map<std::string, Ticker>      tickers_;
    std::map<std::string, BookCallback>   book_cbs_;
    std::map<std::string, TickerCallback> ticker_cbs_;

    void on_message(const std::string& raw);
    void handle_book_update(const std::string& instrument, const std::string& json);
    void handle_ticker_update(const std::string& instrument, const std::string& json);
    void send_subscribe(const std::string& channel);
};

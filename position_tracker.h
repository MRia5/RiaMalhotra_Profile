#pragma once
#include "order_manager.h"
#include <string>
#include <map>
#include <mutex>

struct Position {
    std::string instrument;
    double      net_qty    = 0.0;   // positive = long, negative = short
    double      avg_entry  = 0.0;   // volume-weighted average entry price
    double      realised_pnl   = 0.0;
    double      unrealised_pnl = 0.0;
    double      mark_price = 0.0;

    bool is_flat() const { return std::abs(net_qty) < 1e-9; }
};

/**
 * PositionTracker
 * ---------------
 * Maintains live positions and PnL from order fills.
 * Call update_mark() whenever a new mark price arrives from the feed.
 */
class PositionTracker {
public:
    PositionTracker() = default;

    // Called by OrderManager on fill events
    void on_fill(const Order& order);

    // Update mark price → recalculates unrealised PnL
    void update_mark(const std::string& instrument, double mark_price);

    // Snapshot (thread-safe)
    Position get_position(const std::string& instrument) const;

    // Total unrealised PnL across all positions
    double total_unrealised_pnl() const;

    // Total realised PnL
    double total_realised_pnl() const;

    // Print all positions to stdout
    void print_summary() const;

    // Risk check: returns true if any position exceeds max_qty
    bool check_limits(double max_qty) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, Position> positions_;

    void update_position(const std::string& instrument,
                         double qty, double price);
};

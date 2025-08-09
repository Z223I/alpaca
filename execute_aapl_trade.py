#!/usr/bin/env python3
"""
Execute a live trade for AAPL using the trade generator for Bruce paper account.
"""
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def execute_aapl_trade():
    """Execute AAPL trade using TradeGenerator for Bruce paper account."""
    
    print("🚀 Executing AAPL Trade via TradeGenerator")
    print("=" * 50)
    print("Account: Bruce/paper")
    print("Symbol: AAPL")
    print("Amount: $1000")
    print("Strategy: Buy-Market-Trailing-Sell-Take-Profit-Percent")
    print("Trailing %: 7.5%")
    print("Take Profit %: 15.0%")
    print()
    
    try:
        from atoms.alerts.trade_generator import TradeGenerator
        
        # Initialize trade generator
        trades_dir = Path(__file__).parent / "historical_data"
        generator = TradeGenerator(trades_dir, test_mode=False)  # Live mode
        
        print(f"✓ TradeGenerator initialized")
        print(f"  Max trades per session: {generator.max_trades_per_session}")
        print(f"  Test mode: {generator.test_mode}")
        print()
        
        # Create a mock superduper alert for AAPL
        superduper_alert = {
            "symbol": "AAPL",
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S-0400'),
            "alert_type": "superduper_alert",
            "alert_message": "🎯🎯 **SUPERDUPER ALERT** 🎯🎯\\n\\n🚀📈 **AAPL** @ **Current Price**\\n📊 **STRONG UPTREND** | 🔥 **VERY STRONG**\\n\\n🎯 **Manual Trade Execution**\\n⚡ **Alert Level:** HIGH\\n⚠️ **Risk Level:** LOW"
        }
        
        print("✓ Created mock superduper alert for AAPL")
        print(f"  Alert type: {superduper_alert['alert_type']}")
        print()
        
        # Load account configuration
        config = generator.load_account_config()
        if config is None:
            print("❌ Failed to load account configuration")
            return False
        
        # Get Bruce paper account configuration    
        bruce_paper_config = config.get_environment_config("alpaca", "Bruce", "paper")
        
        print("✓ Loaded Bruce/paper account configuration:")
        print(f"  auto_trade: {bruce_paper_config.auto_trade}")
        print(f"  auto_amount: {bruce_paper_config.auto_amount}")
        print(f"  trailing_percent: {bruce_paper_config.trailing_percent}")
        print(f"  take_profit_percent: {bruce_paper_config.take_profit_percent}")
        print()
        
        # Use $1000 as requested (override the default auto_amount)
        trade_amount = 1000
        print(f"✓ Using trade amount: ${trade_amount} (overriding default ${bruce_paper_config.auto_amount})")
        print()
        
        # Create trade record
        trade_record = generator.create_trade_record(
            superduper_alert, 
            "Bruce", 
            "paper",
            trade_amount,  # Use $1000 as requested
            bruce_paper_config.trailing_percent,
            bruce_paper_config.take_profit_percent
        )
        
        if trade_record is None:
            print("❌ Failed to create trade record")
            return False
            
        print("✓ Created trade record:")
        print(f"  Symbol: {trade_record['symbol']}")
        print(f"  Amount: ${trade_record['auto_amount']}")
        print(f"  Trailing %: {trade_record['trailing_percent']}")
        print(f"  Take Profit %: {trade_record['take_profit_percent']}")
        print(f"  Command: {trade_record['trade_parameters']['command_type']}")
        print()
        
        # Execute the trade
        print("🚀 Executing trade command...")
        print("🔥 THIS WILL PLACE REAL ORDERS IN YOUR PAPER ACCOUNT! 🔥")
        print("Strategy: Market Buy → Trailing Sell (7.5%) → Take Profit (15%)")
        print()
        
        # Show the exact command that will be executed
        print("Command to execute:")
        print(f"  python code/alpaca.py --buy-market-trailing-sell-take-profit-percent \\")
        print(f"    --symbol AAPL --amount 1000 --trailing-percent 7.5 \\")
        print(f"    --take-profit-percent 15.0 --submit --account-name Bruce --account paper")
        print()
        
        # Execute trade command
        updated_trade_record = generator.execute_trade_command(trade_record)
        
        # Show results
        execution_status = updated_trade_record.get('execution_status', {})
        success = execution_status.get('success', 'unknown')
        
        print("📊 Trade Execution Results:")
        print("=" * 30)
        print(f"Success: {success}")
        print(f"Initiated: {execution_status.get('initiated', False)}")
        print(f"Completed: {execution_status.get('completed', False)}")
        print(f"Execution Time: {execution_status.get('execution_time', 'N/A')}")
        print(f"Return Code: {execution_status.get('return_code', 'N/A')}")
        print()
        
        if execution_status.get('stdout'):
            print("📤 Command Output:")
            print(execution_status['stdout'])
            print()
            
        if execution_status.get('stderr'):
            print("⚠️ Command Errors:")
            print(execution_status['stderr'])
            print()
        
        # Save and show file location
        trade_filename = f"trade_{trade_record['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_trade_record(updated_trade_record, trade_filename)
        
        print(f"💾 Trade record saved: {trades_dir / trade_filename}")
        
        if success == "yes":
            print("🎉 Trade executed successfully!")
            return True
        else:
            print("❌ Trade execution failed or encountered issues")
            return False
            
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = execute_aapl_trade()
    sys.exit(0 if success else 1)
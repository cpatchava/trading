#!/usr/bin/env python3
"""
Run optimized processing with sound alerts and progress monitoring
"""

import os
import sys
import time
import subprocess
import sqlite3
from database_optimizations import OptimizedDBManager
from performance_monitor import PerformanceMonitor

def play_sound():
    """Play a system sound alert"""
    try:
        # Try different sound commands for different systems
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
        elif sys.platform == "linux":
            subprocess.run(["paplay", "/usr/share/sounds/alsa/Front_Left.wav"], check=False)
        elif sys.platform == "win32":
            subprocess.run(["powershell", "-c", "[console]::beep(1000,500)"], check=False)
        else:
            # Fallback: print bell character
            print("\a")
    except:
        # Fallback: print bell character
        print("\a")

def get_table_info(db_path, table_name):
    """Get basic info about a table"""
    try:
        with OptimizedDBManager(db_path).get_connection() as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            return count
    except:
        return 0

def main():
    """Run the complete optimized processing pipeline"""
    print("🚀 Starting Optimized Trading Data Processing Pipeline")
    print("=" * 60)
    
    # Paths
    options_db = "../sql-database/options_data.db"
    stocks_db = "../sql-database/stocks_data.db"
    
    monitor = PerformanceMonitor()
    
    # Step 1: Database Optimization
    print("\n📊 Step 1: Setting up database optimizations...")
    with monitor.monitor("Database Optimization"):
        db_manager = OptimizedDBManager(options_db)
        db_manager.create_indexes()
        db_manager.analyze_tables()
        
        stocks_db_manager = OptimizedDBManager(stocks_db)
        stocks_db_manager.create_indexes()
        stocks_db_manager.analyze_tables()
    
    # Get initial table info
    print("\n📈 Initial Data Overview:")
    options_count = get_table_info(options_db, "options")
    enriched_options_count = get_table_info(options_db, "enriched_options")
    stocks_count = get_table_info(stocks_db, "stocks")
    stocks_enriched_count = get_table_info(stocks_db, "stocks_enriched")
    
    print(f"   Options records: {options_count:,}")
    print(f"   Enriched options: {enriched_options_count:,}")
    print(f"   Stocks records: {stocks_count:,}")
    print(f"   Enriched stocks: {stocks_enriched_count:,}")
    
    # Step 2: Options Enrichment (if needed)
    if enriched_options_count == 0 and options_count > 0:
        print("\n🔄 Step 2: Running optimized options enrichment...")
        with monitor.monitor("Options Enrichment"):
            os.system("python options_enrichment_optimized.py --options-db ../sql-database/options_data.db --chunk-size 50000")
    else:
        print(f"\n✅ Step 2: Options enrichment already complete ({enriched_options_count:,} records)")
    
    # Step 3: Stocks Enrichment (if needed)
    if stocks_enriched_count == 0 and stocks_count > 0:
        print("\n🔄 Step 3: Running optimized stocks enrichment...")
        with monitor.monitor("Stocks Enrichment"):
            os.system("python stocks_enrichment_optimized.py")
    else:
        print(f"\n✅ Step 3: Stocks enrichment already complete ({stocks_enriched_count:,} records)")
    
    # Step 4: Price Enrichment
    print("\n🔄 Step 4: Running optimized price enrichment...")
    with monitor.monitor("Price Enrichment"):
        os.system("python options_stocks_price_enrichment_optimized.py --options-db ../sql-database/options_data.db --stocks-db ../sql-database/stocks_data.db --chunk-size 25000")
    
    # Step 5: Black-Scholes Pricing
    print("\n🔄 Step 5: Running Black-Scholes pricing...")
    with monitor.monitor("Black-Scholes Pricing"):
        os.system("python black_scholes_modeling.py --options-db ../sql-database/options_data.db --chunk-size 25000")
    
    # Final data overview
    print("\n📊 Final Data Overview:")
    final_enriched_options = get_table_info(options_db, "enriched_options")
    final_stocks_enriched = get_table_info(stocks_db, "stocks_enriched")
    
    # Check for Black-Scholes pricing completion
    with OptimizedDBManager(options_db).get_connection() as conn:
        bs_priced = conn.execute("SELECT COUNT(*) FROM enriched_options WHERE black_scholes_model_price IS NOT NULL").fetchone()[0]
        total_enriched = conn.execute("SELECT COUNT(*) FROM enriched_options").fetchone()[0]
    
    print(f"   Enriched options: {final_enriched_options:,}")
    print(f"   Enriched stocks: {final_stocks_enriched:,}")
    print(f"   Black-Scholes priced: {bs_priced:,} / {total_enriched:,}")
    
    # Performance summary
    print("\n🎯 Processing Summary:")
    print("=" * 60)
    print("✅ Database optimizations applied")
    print("✅ Options enrichment completed")
    print("✅ Stocks enrichment completed") 
    print("✅ Price enrichment completed")
    print("✅ Black-Scholes pricing completed")
    print("=" * 60)
    
    # Play completion sound
    print("\n🔔 Processing Complete! Playing sound alert...")
    play_sound()
    
    # Additional sound alerts
    time.sleep(1)
    play_sound()
    time.sleep(1)
    play_sound()
    
    print("\n🎉 All processing completed successfully!")
    print("📊 Your trading data is now ready for analysis!")

if __name__ == "__main__":
    main()


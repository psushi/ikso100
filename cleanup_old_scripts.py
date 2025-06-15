#!/usr/bin/env python3
"""
Cleanup Script for Redundant Data Collection Files
Removes old collection scripts and keeps only the unified system
"""

import os
import sys
from pathlib import Path


def cleanup_redundant_scripts():
    """Remove redundant data collection scripts"""
    
    # Scripts to remove (keeping only unified_data_collector.py)
    scripts_to_remove = [
        "collect_dataset_enhanced_trajectories.py",
        "collect_dataset_manual.py", 
        "collect_dataset_manual_v2.py",
        "collect_dataset_with_viewer.py",
        "demo_collect_with_trajectories.py",
        "demo_enhanced_collection.py",
        "demo_quick_trajectory_collection.py",
        "demo_trajectories.py",
        "demo_trajectories_quick.py"
    ]
    
    print("🧹 Cleaning up redundant data collection scripts...")
    print("=" * 60)
    
    removed_count = 0
    for script in scripts_to_remove:
        if os.path.exists(script):
            try:
                os.remove(script)
                print(f"✅ Removed: {script}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {script}: {e}")
        else:
            print(f"⚪ Not found: {script}")
    
    print("=" * 60)
    print(f"🎉 Cleanup completed! Removed {removed_count} redundant scripts")
    print("\n📋 Remaining data collection system:")
    print("   • unified_data_collector.py - Main collection script")
    print("   • validate_lerobot_dataset.py - Dataset validation")
    print("   • main.py - Core simulation (unchanged)")
    print("   • aruco_detection.py - ArUco integration (unchanged)")
    print("   • calibrate.py - Camera calibration (unchanged)")
    
    return removed_count


def main():
    """Main function"""
    print("🤖 SO-100 Arm Project Cleanup")
    
    try:
        response = input("Remove redundant data collection scripts? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("👋 Cleanup cancelled")
            return 0
    except KeyboardInterrupt:
        print("\n👋 Cleanup cancelled")
        return 0
    
    removed_count = cleanup_redundant_scripts()
    
    if removed_count > 0:
        print(f"\n✅ Successfully cleaned up {removed_count} redundant scripts")
        print("🚀 Use 'python unified_data_collector.py' for all data collection needs")
    else:
        print("\n⚪ No files were removed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
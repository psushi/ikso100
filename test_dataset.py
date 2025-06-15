#!/usr/bin/env python3
"""
Test script for dataset collection
Runs a short test to verify everything works correctly
"""

from collect_dataset import collect_dataset_episodes

if __name__ == "__main__":
    print("Testing dataset collection with 1 episode, 10 seconds duration...")
    
    try:
        # Test with minimal parameters
        collect_dataset_episodes(num_episodes=1, episode_duration=10.0)
        print("✅ Dataset collection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 
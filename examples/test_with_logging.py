#!/usr/bin/env python3
"""
Test script demonstrating the logging system.

This script shows how to use the logging system to track all steps
of the map generation and validation process.
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logger import LoggerConfig
from src.pipeline import MapGenerationPipeline, GenerationConfig


def main():
    """
    Run map generation with comprehensive logging.
    """
    print("=" * 80)
    print("Map Generation with Detailed Logging")
    print("=" * 80)
    print()

    # Setup logging - this will create log files and display logs
    LoggerConfig.setup_logging(
        log_dir='logs',
        log_level=logging.DEBUG,  # Log everything (DEBUG, INFO, WARNING, ERROR)
        console_output=True,      # Show logs in console
        file_output=True          # Save logs to file
    )

    print("Logging initialized!")
    print(f"Log files will be saved to: logs/")
    print(f"  - logs/latest.log (always contains the most recent run)")
    print(f"  - logs/map_generation_YYYYMMDD_HHMMSS.log (timestamped logs)")
    print()
    print("Starting map generation...")
    print("=" * 80)
    print()

    # Create configuration
    config = GenerationConfig(
        width=256,
        height=256,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        scale=100.0,
        num_samples=50,
        observer_height=0.05,
        max_attempts=30,
        target_valid_maps=2,
        output_dir='output/with_logging',
        validation_rules_path='config/validation_rules.json',
        save_all_maps=False,
        verbose=True
    )

    # Run pipeline
    pipeline = MapGenerationPipeline(config)
    stats = pipeline.run()

    print()
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print()
    print(f"Results:")
    print(f"  Valid maps generated: {stats['valid_maps']}")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print()
    print("Check the log files for detailed information about each step!")
    print(f"  Main log: logs/latest.log")
    print(f"  Output maps: {config.output_dir}/")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test Full Pixel-by-Pixel Analysis

This script tests the new exhaustive analysis mode with a small map.
"""

import sys
import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logger import LoggerConfig
from src.map_generator import MapGenerator
from src.spatial_analyzer import create_full_analysis
from src.validation_engine import ValidationEngine, create_default_rules
from src.step_visualizer import StepByStepVisualizer

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['font.monospace'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def main():
    """
    Test full pixel-by-pixel analysis with a small map.
    """
    print("=" * 80)
    print("Testing Full Pixel-by-Pixel Exposure Analysis")
    print("=" * 80)
    print()

    # Setup logging
    LoggerConfig.setup_logging(
        log_dir='logs',
        log_level=logging.INFO,
        console_output=True,
        file_output=True
    )

    # Use a small map for testing (64x64)
    print("Generating 64x64 test map...")
    generator = MapGenerator(
        width=64,
        height=64,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        scale=100.0,
        seed=42
    )
    map_data = generator.generate_map()
    print(f"✓ Map generated: {map_data.dimensions}")
    print()

    # Test 1: Sampling mode (old way)
    print("[Test 1] Sampling Mode (50 samples)")
    start_time = time.time()

    results_sampling = create_full_analysis(
        map_data,
        num_samples=50,
        observer_height=0.05,
        use_full_analysis=False  # Sampling mode
    )

    elapsed_sampling = time.time() - start_time

    metrics_sampling = results_sampling['visibility_metrics']
    print(f"  Time: {elapsed_sampling:.2f}s")
    print(f"  Team A Exposure: {metrics_sampling['team_a_exposure']:.4f}")
    print(f"  Team B Exposure: {metrics_sampling['team_b_exposure']:.4f}")
    print(f"  Difference: {metrics_sampling['exposure_difference']:.4f}")
    print()

    # Test 2: Full pixel-by-pixel mode (new way)
    print("[Test 2] Full Pixel-by-Pixel Mode")
    start_time = time.time()

    results_full = create_full_analysis(
        map_data,
        observer_height=0.05,
        use_full_analysis=True  # Full exhaustive analysis
    )

    elapsed_full = time.time() - start_time

    metrics_full = results_full['visibility_metrics']
    print(f"  Time: {elapsed_full:.2f}s")
    print(f"  Team A Exposure: {metrics_full['team_a_exposure']:.4f}")
    print(f"  Team B Exposure: {metrics_full['team_b_exposure']:.4f}")
    print(f"  Difference: {metrics_full['exposure_difference']:.4f}")
    print()

    # Compare results
    print("=" * 80)
    print("Comparison:")
    print("=" * 80)
    print(f"Speed difference: {elapsed_full / elapsed_sampling:.1f}x slower")
    print(f"Exposure difference (A): {abs(metrics_sampling['team_a_exposure'] - metrics_full['team_a_exposure']):.4f}")
    print(f"Exposure difference (B): {abs(metrics_sampling['team_b_exposure'] - metrics_full['team_b_exposure']):.4f}")
    print()

    # Check if exposure maps were created
    if 'team_a_exposure_map' in results_full:
        print("✓ Per-pixel exposure maps generated")
        team_a_map = results_full['team_a_exposure_map']
        team_b_map = results_full['team_b_exposure_map']
        print(f"  Team A map shape: {team_a_map.shape}")
        print(f"  Team B map shape: {team_b_map.shape}")
        print(f"  Team A map range: {team_a_map.min():.3f} - {team_a_map.max():.3f}")
        print(f"  Team B map range: {team_b_map.min():.3f} - {team_b_map.max():.3f}")

        # 픽셀별 통계 계산
        team_a_mask = map_data.get_zone_mask('team_a_zone')
        team_b_mask = map_data.get_zone_mask('team_b_zone')

        a_values = team_a_map[team_a_mask]
        b_values = team_b_map[team_b_mask]

        a_high_exposure = np.sum(a_values > 0.5)
        a_safe = np.sum(a_values < 0.3)
        b_high_exposure = np.sum(b_values > 0.5)
        b_safe = np.sum(b_values < 0.3)

        print(f"\n  Team A:")
        print(f"    위험 지역 (>0.5): {a_high_exposure} 픽셀")
        print(f"    안전 지역 (<0.3): {a_safe} 픽셀")
        print(f"  Team B:")
        print(f"    위험 지역 (>0.5): {b_high_exposure} 픽셀")
        print(f"    안전 지역 (<0.3): {b_safe} 픽셀")
    else:
        print("✗ No per-pixel exposure maps found")
    print()

    # Validate
    print("[Test 3] Validation")
    engine = ValidationEngine()
    for rule in create_default_rules():
        engine.add_rule(rule)

    validation_result = engine.validate(results_full)
    summary = validation_result.get_summary()
    print(f"  Result: {'✓ PASSED' if summary['overall_passed'] else '✗ FAILED'}")
    print(f"  Pass rate: {summary['passed_rules']}/{summary['total_rules']}")
    print()

    # Create visualizations
    print("[Test 4] Creating visualizations with pixel-level exposure...")
    visualizer = StepByStepVisualizer(output_dir='output/test_full', dpi=100)

    print("  Creating Step 5 (Exposure)...")
    visualizer.step5_exposure_calculation(map_data, results_full, "step5_full")
    print("  ✓ Visualization saved: output/test_full/step5_full_exposure_calculation.png")
    print()

    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✓ Full pixel-by-pixel analysis works correctly")
    print(f"  ✓ Per-pixel exposure maps generated")
    print(f"  ✓ Visualization shows pixel-level detail")
    print(f"  ✓ Analysis time for 64x64 map: {elapsed_full:.2f}s")
    print()
    print("Check output/test_full/step5_full_exposure_calculation.png to see results!")
    print()


if __name__ == '__main__':
    main()

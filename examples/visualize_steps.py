#!/usr/bin/env python3
"""
Step-by-Step Visualization Demo

This script generates a map and creates detailed visualizations for each step
of the generation and validation process.
"""

import sys
import os
import logging
import time
import matplotlib.pyplot as plt

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
    Generate a map and create step-by-step visualizations.
    """
    print("=" * 80)
    print("Step-by-Step Map Generation Visualization")
    print("=" * 80)
    print()

    # Setup logging
    LoggerConfig.setup_logging(
        log_dir='logs',
        log_level=logging.INFO,
        console_output=False,  # Don't clutter console
        file_output=True
    )

    print("Generating map and creating visualizations...")
    print()

    # 맵 파라미터 설정
    MAP_WIDTH = 64
    MAP_HEIGHT = 64
    USE_FULL_ANALYSIS = True

    print(f"Map Size: {MAP_WIDTH}x{MAP_HEIGHT}")
    print(f"Analysis Mode: {'FULL (전수 조사)' if USE_FULL_ANALYSIS else 'SAMPLING'}")
    if USE_FULL_ANALYSIS:
        print(f"⚠️  예상 시간: 약 2-5분")
    print()

    # Step 1: Generate map
    print("[1/6] Generating heightmap with Perlin noise...")
    start_time = time.time()
    generator = MapGenerator(
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        scale=100.0,
        seed=42  # Fixed seed for reproducibility
    )
    map_data = generator.generate_map()
    elapsed = time.time() - start_time
    print(f"     ✓ Heightmap generated ({elapsed:.2f}s)")

    # Step 2: Analyze (using FULL pixel-by-pixel analysis)
    print("[2/6] Performing FULL pixel-by-pixel spatial analysis...")
    if USE_FULL_ANALYSIS:
        print("     (전수 조사 모드 - 모든 픽셀 분석 중...)")
    start_time = time.time()
    analysis_results = create_full_analysis(
        map_data,
        observer_height=0.05,
        use_full_analysis=USE_FULL_ANALYSIS  # Use exhaustive pixel-by-pixel analysis
    )
    elapsed = time.time() - start_time
    metrics = analysis_results['visibility_metrics']
    print(f"     ✓ Analysis complete ({elapsed:.1f}s = {elapsed/60:.2f}분)")
    print(f"       Team A Exposure: {metrics['team_a_exposure']:.4f} ({metrics['team_a_exposure']*100:.2f}%)")
    print(f"       Team B Exposure: {metrics['team_b_exposure']:.4f} ({metrics['team_b_exposure']*100:.2f}%)")
    print(f"       Difference: {metrics['exposure_difference']:.4f}")

    # 픽셀별 통계 출력
    if 'team_a_exposure_map' in analysis_results:
        team_a_map = analysis_results['team_a_exposure_map']
        team_b_map = analysis_results['team_b_exposure_map']
        team_a_mask = map_data.get_zone_mask('team_a_zone')
        team_b_mask = map_data.get_zone_mask('team_b_zone')

        a_values = team_a_map[team_a_mask]
        b_values = team_b_map[team_b_mask]

        print(f"       Team A 노출도 범위: {a_values.min():.3f} ~ {a_values.max():.3f}")
        print(f"       Team B 노출도 범위: {b_values.min():.3f} ~ {b_values.max():.3f}")

    # Step 3: Validate
    print("[3/6] Validating against rules...")
    engine = ValidationEngine()
    for rule in create_default_rules():
        engine.add_rule(rule)

    validation_result = engine.validate(analysis_results)
    summary = validation_result.get_summary()
    print(f"     ✓ Validation complete: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
    print(f"       {summary['passed_rules']}/{summary['total_rules']} rules passed")

    # Step 4-6: Create visualizations
    print("[4/6] Creating step-by-step visualizations...")
    visualizer = StepByStepVisualizer(output_dir='output/steps', dpi=100)

    print("     Creating Step 1: Raw heightmap...")
    visualizer.step1_raw_heightmap(map_data)

    print("     Creating Step 2: Team zones...")
    visualizer.step2_team_zones(map_data)

    print("     Creating Step 3: Sample points...")
    visualizer.step3_sample_points(map_data, num_samples=50)

    print("     Creating Step 4: Line-of-sight demo...")
    visualizer.step4_line_of_sight_demo(map_data, num_demo_lines=10)

    print("     Creating Step 5: Exposure calculation...")
    visualizer.step5_exposure_calculation(map_data, analysis_results)

    print("     Creating Step 6: Validation results...")
    visualizer.step6_validation_results(validation_result)

    print("     ✓ All visualizations created")

    print()
    print("=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  output/steps/step1_raw_heightmap.png       - Raw Perlin noise terrain")
    print("  output/steps/step2_team_zones.png          - Team zone assignment")
    print("  output/steps/step3_sample_points.png       - Sample point distribution")
    print("  output/steps/step4_line_of_sight.png       - Line-of-sight calculations")
    print("  output/steps/step5_exposure_calculation.png - Exposure analysis")
    print("  output/steps/step6_validation_results.txt  - Validation results report")
    print()
    print("Open the images and text report to see each step of the process!")
    print()


def generate_multiple_examples():
    """
    Generate visualizations for multiple maps to show variety.
    """
    print("=" * 80)
    print("Generating Multiple Map Examples")
    print("=" * 80)
    print()

    # Setup logging (silent)
    LoggerConfig.setup_logging(
        log_dir='logs',
        log_level=logging.WARNING,
        console_output=False,
        file_output=True
    )

    # 전수 조사용 작은 맵 크기
    MAP_WIDTH = 64
    MAP_HEIGHT = 64

    seeds = [42, 123, 456, 789]  # Different seeds for variety

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] Generating map with seed {seed}...")
        start_time = time.time()

        # Generate map
        generator = MapGenerator(width=MAP_WIDTH, height=MAP_HEIGHT, seed=seed)
        map_data = generator.generate_map()

        # Analyze (FULL pixel-by-pixel)
        analysis_results = create_full_analysis(
            map_data,
            observer_height=0.05,
            use_full_analysis=True  # 전수 조사 모드
        )

        # Validate
        engine = ValidationEngine()
        for rule in create_default_rules():
            engine.add_rule(rule)
        validation_result = engine.validate(analysis_results)

        # Create visualizations
        visualizer = StepByStepVisualizer(output_dir=f'output/steps_example_{i}', dpi=80)
        visualizer.create_all_steps(map_data, analysis_results, validation_result,
                                   num_samples=50, step_prefix=f"map{i}_step")

        elapsed = time.time() - start_time
        summary = validation_result.get_summary()
        metrics = analysis_results['visibility_metrics']
        status = "✓ PASSED" if summary['overall_passed'] else "✗ FAILED"
        print(f"     {status} - Time: {elapsed:.1f}s - Exposure diff: {metrics['exposure_difference']:.4f}")
        print(f"     → output/steps_example_{i}/")

    print()
    print("=" * 80)
    print("Multiple examples generated!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--multiple':
        generate_multiple_examples()
    else:
        main()

        print("TIP: Run with --multiple flag to generate multiple examples:")
        print("     python visualize_steps.py --multiple")
        print()

#!/usr/bin/env python3
"""
Example script demonstrating how to use the Procedural Map Validation System.

This script shows various ways to generate and validate maps for competitive
multiplayer games with balanced sight exposure.
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import MapGenerationPipeline, GenerationConfig, quick_generate
from src.map_generator import MapGenerator
from src.spatial_analyzer import create_full_analysis
from src.validation_engine import ValidationEngine, create_default_rules
from src.visualization import MapVisualizer, save_analysis_json


def example_1_quick_generation():
    """
    Example 1: Quick generation with default settings.

    This is the simplest way to generate valid maps.
    """
    print("=" * 70)
    print("Example 1: Quick Generation")
    print("=" * 70)

    stats = quick_generate(
        num_maps=2,
        width=256,
        height=256,
        output_dir='output/example1',
        max_attempts=50
    )

    print(f"\nGeneration statistics: {stats}")


def example_2_custom_configuration():
    """
    Example 2: Using custom configuration for fine-tuned control.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)

    # Create custom configuration
    config = GenerationConfig(
        width=512,
        height=512,
        octaves=8,  # More detail
        persistence=0.6,
        lacunarity=2.5,
        scale=150.0,
        num_samples=150,  # More accurate analysis
        observer_height=0.05,
        max_attempts=100,
        target_valid_maps=3,
        output_dir='output/example2',
        validation_rules_path='config/validation_rules.json',
        save_all_maps=False,
        verbose=True
    )

    # Run pipeline
    pipeline = MapGenerationPipeline(config)
    stats = pipeline.run()

    print(f"\nFinal statistics: {stats}")


def example_3_single_map_workflow():
    """
    Example 3: Step-by-step workflow for a single map.

    This demonstrates the individual components of the system.
    """
    print("\n" + "=" * 70)
    print("Example 3: Single Map Step-by-Step Workflow")
    print("=" * 70)

    # Step 1: Generate a map
    print("\n[1/5] Generating terrain...")
    generator = MapGenerator(
        width=256,
        height=256,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        scale=100.0,
        seed=12345  # Fixed seed for reproducibility
    )
    map_data = generator.generate_map()
    print(f"✓ Generated {map_data.dimensions[0]}x{map_data.dimensions[1]} map")
    print(f"  Zones: {list(map_data.zones.keys())}")

    # Step 2: Analyze spatial properties
    print("\n[2/5] Analyzing visibility and spatial properties...")
    analysis_results = create_full_analysis(
        map_data,
        num_samples=100,
        observer_height=0.05
    )

    metrics = analysis_results['visibility_metrics']
    print(f"✓ Analysis complete")
    print(f"  Team A Exposure: {metrics.get('team_a_exposure', 0):.3f}")
    print(f"  Team B Exposure: {metrics.get('team_b_exposure', 0):.3f}")
    print(f"  Exposure Difference: {metrics.get('exposure_difference', 0):.3f}")

    # Step 3: Validate against rules
    print("\n[3/5] Validating against rules...")
    validation_engine = ValidationEngine()
    for rule in create_default_rules():
        validation_engine.add_rule(rule)

    validation_result = validation_engine.validate(analysis_results)
    summary = validation_result.get_summary()

    print(f"✓ Validation complete")
    print(f"  Result: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
    print(f"  Passed Rules: {summary['passed_rules']}/{summary['total_rules']}")

    if not summary['overall_passed']:
        print("\n  Failed rules:")
        for detail in summary['failed_rule_details']:
            print(f"    - {detail['rule_id']}: {detail['message']}")

    # Step 4: Generate visualizations
    print("\n[4/5] Creating visualizations...")
    os.makedirs('output/example3', exist_ok=True)
    visualizer = MapVisualizer()

    visualizer.save_heightmap(map_data, 'output/example3/heightmap.png')
    visualizer.visualize_zones(map_data, 'output/example3/zones.png')
    visualizer.visualize_exposure_heatmap(
        map_data,
        analysis_results,
        'output/example3/exposure_heatmap.png'
    )
    visualizer.create_analysis_report(
        map_data,
        analysis_results,
        validation_result,
        'output/example3/full_report.png'
    )

    print("✓ Visualizations saved to output/example3/")

    # Step 5: Save analysis data
    print("\n[5/5] Saving analysis data...")
    save_analysis_json(
        analysis_results,
        validation_result,
        'output/example3/analysis_data.json'
    )
    print("✓ JSON data saved")

    print("\n" + "=" * 70)
    print("Example 3 Complete!")
    print("=" * 70)


def example_4_batch_generation():
    """
    Example 4: Batch generation with progress tracking.
    """
    print("\n" + "=" * 70)
    print("Example 4: Batch Generation with Progress Tracking")
    print("=" * 70)

    def progress_callback(valid_count, attempt_count):
        """Custom progress callback."""
        if attempt_count % 5 == 0:
            print(f"  Progress: {valid_count} valid maps found after {attempt_count} attempts")

    config = GenerationConfig(
        width=256,
        height=256,
        target_valid_maps=5,
        max_attempts=100,
        output_dir='output/example4',
        num_samples=75,
        verbose=False  # Disable built-in verbose output
    )

    print(f"\nGenerating {config.target_valid_maps} valid maps...")
    print(f"Max attempts: {config.max_attempts}")

    pipeline = MapGenerationPipeline(config)
    stats = pipeline.run(progress_callback=progress_callback)

    print(f"\n✓ Batch generation complete!")
    print(f"  Valid maps: {stats['valid_maps']}")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total time: {stats['total_time']:.2f}s")


def example_5_custom_rules():
    """
    Example 5: Creating and using custom validation rules.
    """
    print("\n" + "=" * 70)
    print("Example 5: Custom Validation Rules")
    print("=" * 70)

    from src.validation_engine import ValidationRule

    # Create custom rules for testing/development
    custom_rules = [
        ValidationRule(
            rule_id='lenient_max_exposure',
            metric_name='max_exposure',
            operator='less_than',
            threshold=0.6,  # More lenient than default
            description='Lenient maximum exposure for testing'
        ),
        ValidationRule(
            rule_id='lenient_balance',
            metric_name='exposure_difference',
            operator='less_than',
            threshold=0.25,  # More lenient
            description='Lenient exposure balance'
        )
    ]

    # Create engine with custom rules
    validation_engine = ValidationEngine(custom_rules)

    # Save custom rules for reuse
    os.makedirs('output/example5', exist_ok=True)
    validation_engine.save_rules_to_file('output/example5/custom_rules.json')
    print("✓ Custom rules saved to output/example5/custom_rules.json")

    # Use custom rules in pipeline
    config = GenerationConfig(
        width=256,
        height=256,
        target_valid_maps=2,
        max_attempts=20,
        output_dir='output/example5',
        validation_rules_path='output/example5/custom_rules.json',
        verbose=True
    )

    pipeline = MapGenerationPipeline(config)
    stats = pipeline.run()

    print(f"\nGeneration with custom rules complete!")
    print(f"Success rate with lenient rules: {stats['success_rate']:.1%}")


def main():
    """
    Run all examples or specific ones based on command line arguments.
    """
    print("\n" + "=" * 70)
    print("Procedural Map Validation System - Examples")
    print("=" * 70)

    # Check command line arguments
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_1_quick_generation,
            '2': example_2_custom_configuration,
            '3': example_3_single_map_workflow,
            '4': example_4_batch_generation,
            '5': example_5_custom_rules
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"\nError: Example {example_num} not found.")
            print("Available examples: 1, 2, 3, 4, 5")
            print("\nUsage: python generate_maps.py [example_number]")
            print("       python generate_maps.py          # Run all examples")
    else:
        # Run all examples
        print("\nRunning all examples...")
        print("This may take a few minutes depending on your hardware.")
        print("\nYou can also run individual examples:")
        print("  python generate_maps.py 1  # Quick generation")
        print("  python generate_maps.py 2  # Custom configuration")
        print("  python generate_maps.py 3  # Step-by-step workflow")
        print("  python generate_maps.py 4  # Batch generation")
        print("  python generate_maps.py 5  # Custom rules")

        try:
            example_1_quick_generation()
            example_2_custom_configuration()
            example_3_single_map_workflow()
            example_4_batch_generation()
            example_5_custom_rules()

            print("\n" + "=" * 70)
            print("All examples completed successfully!")
            print("Check the 'output/' directory for generated maps and analysis.")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            sys.exit(1)


if __name__ == '__main__':
    main()

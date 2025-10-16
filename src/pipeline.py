"""
Pipeline Module

Orchestrates the complete map generation and validation workflow.
"""

import os
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .map_generator import MapGenerator, MapData
from .spatial_analyzer import create_full_analysis
from .validation_engine import ValidationEngine, ValidationResult
from .visualization import MapVisualizer, save_analysis_json
from .logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


class GenerationConfig:
    """
    Configuration parameters for map generation pipeline.
    """

    def __init__(self,
                 width: int = 512,
                 height: int = 512,
                 octaves: int = 6,
                 persistence: float = 0.5,
                 lacunarity: float = 2.0,
                 scale: float = 100.0,
                 num_samples: int = 100,
                 observer_height: float = 0.05,
                 max_attempts: int = 100,
                 target_valid_maps: int = 1,
                 output_dir: str = 'output',
                 validation_rules_path: Optional[str] = None,
                 save_all_maps: bool = False,
                 verbose: bool = True):
        """
        Initialize generation configuration.

        Args:
            width: Map width in pixels
            height: Map height in pixels
            octaves: Perlin noise octaves
            persistence: Noise persistence factor
            lacunarity: Noise lacunarity factor
            scale: Noise scale
            num_samples: Number of sample points for visibility analysis
            observer_height: Observer height offset
            max_attempts: Maximum generation attempts before giving up
            target_valid_maps: Number of valid maps to generate
            output_dir: Directory for output files
            validation_rules_path: Path to validation rules JSON file
            save_all_maps: Whether to save failed maps for debugging
            verbose: Whether to print progress messages
        """
        self.width = width
        self.height = height
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.scale = scale
        self.num_samples = num_samples
        self.observer_height = observer_height
        self.max_attempts = max_attempts
        self.target_valid_maps = target_valid_maps
        self.output_dir = output_dir
        self.validation_rules_path = validation_rules_path
        self.save_all_maps = save_all_maps
        self.verbose = verbose


class GenerationStats:
    """
    Tracks statistics about the generation process.
    """

    def __init__(self):
        self.attempts = 0
        self.valid_maps = 0
        self.failed_maps = 0
        self.start_time = None
        self.end_time = None
        self.generation_times = []
        self.analysis_times = []

    def start(self):
        """Start timing the generation process."""
        self.start_time = time.time()

    def stop(self):
        """Stop timing the generation process."""
        self.end_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with generation statistics
        """
        avg_gen_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        avg_analysis_time = sum(self.analysis_times) / len(self.analysis_times) if self.analysis_times else 0

        return {
            'total_attempts': self.attempts,
            'valid_maps': self.valid_maps,
            'failed_maps': self.failed_maps,
            'success_rate': self.valid_maps / self.attempts if self.attempts > 0 else 0,
            'total_time': self.get_elapsed_time(),
            'avg_generation_time': avg_gen_time,
            'avg_analysis_time': avg_analysis_time
        }


class MapGenerationPipeline:
    """
    Complete pipeline for generating and validating procedural maps.
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize the pipeline with configuration.

        Args:
            config: GenerationConfig object
        """
        logger.info("=" * 80)
        logger.info("Initializing Map Generation Pipeline")
        logger.info("=" * 80)

        self.config = config
        self.stats = GenerationStats()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output directory: {config.output_dir}")

        # Initialize components
        self.visualizer = MapVisualizer()

        # Setup validation engine
        self.validation_engine = ValidationEngine()
        if config.validation_rules_path and os.path.exists(config.validation_rules_path):
            logger.info(f"Loading validation rules from: {config.validation_rules_path}")
            self.validation_engine.load_rules_from_file(config.validation_rules_path)
        else:
            logger.info("Using default validation rules")
            # Use default rules
            from .validation_engine import create_default_rules
            for rule in create_default_rules():
                self.validation_engine.add_rule(rule)

        logger.info(f"Pipeline initialized with {len(self.validation_engine.rules)} rules")

    def generate_single_map(self, seed: Optional[int] = None) -> MapData:
        """
        Generate a single map.

        Args:
            seed: Optional seed for reproducibility

        Returns:
            MapData object
        """
        generator = MapGenerator(
            width=self.config.width,
            height=self.config.height,
            octaves=self.config.octaves,
            persistence=self.config.persistence,
            lacunarity=self.config.lacunarity,
            scale=self.config.scale,
            seed=seed
        )

        return generator.generate_map()

    def analyze_map(self, map_data: MapData) -> Dict[str, any]:
        """
        Perform spatial analysis on a map.

        Args:
            map_data: MapData to analyze

        Returns:
            Analysis results dictionary
        """
        return create_full_analysis(
            map_data,
            num_samples=self.config.num_samples,
            observer_height=self.config.observer_height
        )

    def validate_map(self, analysis_results: Dict[str, any]) -> ValidationResult:
        """
        Validate map analysis results.

        Args:
            analysis_results: Results from spatial analysis

        Returns:
            ValidationResult object
        """
        return self.validation_engine.validate(analysis_results)

    def save_map_outputs(self,
                        map_data: MapData,
                        analysis_results: Dict[str, any],
                        validation_result: ValidationResult,
                        map_id: str):
        """
        Save all output files for a map.

        Args:
            map_data: MapData object
            analysis_results: Analysis results
            validation_result: Validation result
            map_id: Unique identifier for this map
        """
        base_path = os.path.join(self.config.output_dir, map_id)

        # Save heightmap
        self.visualizer.save_heightmap(
            map_data,
            f"{base_path}_heightmap.png"
        )

        # Save zone visualization
        self.visualizer.visualize_zones(
            map_data,
            f"{base_path}_zones.png"
        )

        # Save exposure heatmap
        self.visualizer.visualize_exposure_heatmap(
            map_data,
            analysis_results,
            f"{base_path}_exposure.png"
        )

        # Save comprehensive report
        self.visualizer.create_analysis_report(
            map_data,
            analysis_results,
            validation_result,
            f"{base_path}_report.png"
        )

        # Save JSON data
        save_analysis_json(
            analysis_results,
            validation_result,
            f"{base_path}_analysis.json"
        )

    def run_single_iteration(self, iteration: int) -> tuple[bool, Optional[MapData], Optional[Dict], Optional[ValidationResult]]:
        """
        Run a single generation iteration.

        Args:
            iteration: Iteration number

        Returns:
            Tuple of (success, map_data, analysis_results, validation_result)
        """
        logger.info(f"--- Iteration {iteration} ---")

        # Generate map
        with PerformanceLogger(logger, "Map generation"):
            map_data = self.generate_single_map()
        gen_time = time.time() - (time.time() - time.time())  # Will be tracked properly by context manager

        # Analyze map
        with PerformanceLogger(logger, "Spatial analysis"):
            analysis_results = self.analyze_map(map_data)

        # Validate
        validation_result = self.validate_map(analysis_results)

        self.stats.attempts += 1

        if validation_result.passed:
            self.stats.valid_maps += 1
            logger.info(f"✓ Iteration {iteration}: Map PASSED validation")
            return True, map_data, analysis_results, validation_result
        else:
            self.stats.failed_maps += 1
            summary = validation_result.get_summary()
            logger.info(f"✗ Iteration {iteration}: Map FAILED validation ({summary['failed_rules']} rules failed)")
            return False, map_data, analysis_results, validation_result

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        """
        Run the complete generation pipeline.

        Args:
            progress_callback: Optional callback function(valid_count, attempt_count)
        """
        self.stats.start()

        logger.info("=" * 80)
        logger.info(f"Starting map generation pipeline")
        logger.info(f"Target: {self.config.target_valid_maps} valid maps")
        logger.info(f"Max attempts: {self.config.max_attempts}")
        logger.info(f"Map size: {self.config.width}x{self.config.height}")
        logger.info(f"Samples per zone: {self.config.num_samples}")
        logger.info("=" * 80)

        if self.config.verbose:
            print(f"Starting map generation pipeline...")
            print(f"Target: {self.config.target_valid_maps} valid maps")
            print(f"Max attempts: {self.config.max_attempts}")
            print(f"Output directory: {self.config.output_dir}")
            print("-" * 60)

        iteration = 0
        while self.stats.valid_maps < self.config.target_valid_maps and iteration < self.config.max_attempts:
            iteration += 1

            success, map_data, analysis_results, validation_result = self.run_single_iteration(iteration)

            if success:
                # Save valid map
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                map_id = f"map_valid_{self.stats.valid_maps:03d}_{timestamp}"

                logger.info(f"Saving valid map: {map_id}")

                if self.config.verbose:
                    print(f"✓ Valid map {self.stats.valid_maps}/{self.config.target_valid_maps} "
                          f"(attempt {iteration}/{self.config.max_attempts})")

                self.save_map_outputs(map_data, analysis_results, validation_result, map_id)

            else:
                if self.config.verbose:
                    summary = validation_result.get_summary()
                    print(f"✗ Attempt {iteration}: Failed {summary['failed_rules']} rule(s)")

                # Optionally save failed maps for debugging
                if self.config.save_all_maps:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    map_id = f"map_failed_{iteration:03d}_{timestamp}"
                    logger.debug(f"Saving failed map for debugging: {map_id}")
                    self.save_map_outputs(map_data, analysis_results, validation_result, map_id)

            # Progress callback
            if progress_callback:
                progress_callback(self.stats.valid_maps, iteration)

        self.stats.stop()

        # Final summary
        summary = self.stats.get_summary()
        logger.info("=" * 80)
        logger.info(f"Pipeline execution complete!")
        logger.info(f"Valid maps: {summary['valid_maps']}/{self.config.target_valid_maps}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Total time: {summary['total_time']:.2f}s")
        logger.info("=" * 80)

        if self.config.verbose:
            print("-" * 60)
            print(f"Generation complete!")
            print(f"Valid maps: {summary['valid_maps']}/{self.config.target_valid_maps}")
            print(f"Success rate: {summary['success_rate']:.1%}")
            print(f"Total time: {summary['total_time']:.2f}s")
            print(f"Avg generation time: {summary['avg_generation_time']:.3f}s")
            print(f"Avg analysis time: {summary['avg_analysis_time']:.3f}s")

        return self.stats.get_summary()


def quick_generate(num_maps: int = 1,
                  width: int = 256,
                  height: int = 256,
                  output_dir: str = 'output',
                  max_attempts: int = 50) -> Dict[str, any]:
    """
    Convenience function for quick map generation with default settings.

    Args:
        num_maps: Number of valid maps to generate
        width: Map width
        height: Map height
        output_dir: Output directory
        max_attempts: Maximum attempts per map

    Returns:
        Generation statistics dictionary
    """
    config = GenerationConfig(
        width=width,
        height=height,
        target_valid_maps=num_maps,
        output_dir=output_dir,
        max_attempts=max_attempts,
        num_samples=50,  # Reduced for speed
        verbose=True
    )

    pipeline = MapGenerationPipeline(config)
    return pipeline.run()

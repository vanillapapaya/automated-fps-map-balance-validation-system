"""
Spatial Analyzer Module

Performs visibility analysis and calculates sight exposure ratios for map validation.
"""

import numpy as np
from typing import Tuple, Dict, List
from .map_generator import MapData
from .logger import get_logger

logger = get_logger(__name__)


class VisibilityAnalyzer:
    """
    Analyzes line-of-sight and calculates sight exposure metrics for map zones.
    """

    def __init__(self, observer_height: float = 0.05, epsilon: float = 0.001):
        """
        Initialize the visibility analyzer.

        Args:
            observer_height: Height offset added to terrain for observer position (in normalized units, 0-1 scale)
            epsilon: Tolerance for floating-point height comparisons
        """
        self.observer_height = observer_height
        self.epsilon = epsilon

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Generate points along a line using Bresenham's algorithm.

        Args:
            x0, y0: Starting point coordinates
            x1, y1: Ending point coordinates

        Returns:
            List of (x, y) coordinate tuples along the line
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def check_line_of_sight(self,
                           observer: Tuple[int, int],
                           target: Tuple[int, int],
                           heightmap: np.ndarray) -> bool:
        """
        Check if observer can see target given terrain elevation.

        Args:
            observer: (x, y) coordinates of observer position
            target: (x, y) coordinates of target position
            heightmap: 2D array of terrain elevations

        Returns:
            True if line of sight is unobstructed, False otherwise
        """
        obs_x, obs_y = observer
        tgt_x, tgt_y = target

        # Get terrain heights
        height, width = heightmap.shape

        # Validate coordinates are within bounds
        if not (0 <= obs_x < width and 0 <= obs_y < height):
            return False
        if not (0 <= tgt_x < width and 0 <= tgt_y < height):
            return False

        obs_elevation = heightmap[obs_y, obs_x] + self.observer_height
        tgt_elevation = heightmap[tgt_y, tgt_x] + self.observer_height

        # Get line points
        line_points = self.bresenham_line(obs_x, obs_y, tgt_x, tgt_y)

        # Skip first and last points (observer and target positions)
        if len(line_points) <= 2:
            return True

        # Calculate total distance for interpolation
        total_distance = np.sqrt((tgt_x - obs_x)**2 + (tgt_y - obs_y)**2)

        if total_distance < self.epsilon:
            return True

        # Check each point along the line
        for i, (px, py) in enumerate(line_points[1:-1], 1):
            # Calculate expected height of unobstructed sightline at this point
            distance_from_obs = np.sqrt((px - obs_x)**2 + (py - obs_y)**2)
            ratio = distance_from_obs / total_distance

            expected_height = obs_elevation + ratio * (tgt_elevation - obs_elevation)
            terrain_height = heightmap[py, px]

            # Check if terrain blocks the view
            if terrain_height > expected_height + self.epsilon:
                return False

        return True

    def calculate_zone_exposure(self,
                                observer_zone: str,
                                target_zone: str,
                                map_data: MapData,
                                num_observer_samples: int = 100,
                                num_target_samples: int = 100) -> float:
        """
        Calculate sight exposure ratio from observer zone to target zone.

        The exposure ratio represents how vulnerable the target zone is to
        observation from the observer zone.

        Args:
            observer_zone: Name of the zone containing observers
            target_zone: Name of the zone being observed
            map_data: MapData object containing heightmap and zones
            num_observer_samples: Number of observation points to sample
            num_target_samples: Number of target points to sample

        Returns:
            Exposure ratio (0-1) where higher means more vulnerable

        Raises:
            KeyError: If zone names don't exist in map_data
        """
        logger.debug(f"Calculating exposure: {observer_zone} -> {target_zone}")

        if observer_zone not in map_data.zones:
            raise KeyError(f"Observer zone '{observer_zone}' not found")
        if target_zone not in map_data.zones:
            raise KeyError(f"Target zone '{target_zone}' not found")

        # Get sample points
        observer_points = map_data.zones[observer_zone].get_sample_points(num_observer_samples)
        target_points = map_data.zones[target_zone].get_sample_points(num_target_samples)

        if len(observer_points) == 0 or len(target_points) == 0:
            return 0.0

        logger.debug(f"Analyzing {len(observer_points)} observer points vs {len(target_points)} target points")

        # Calculate visibility for each observer
        observer_exposure_scores = []

        for obs_point in observer_points:
            visible_count = 0

            for tgt_point in target_points:
                if self.check_line_of_sight(
                    tuple(obs_point),
                    tuple(tgt_point),
                    map_data.heightmap
                ):
                    visible_count += 1

            # Exposure score for this observer position
            exposure_score = visible_count / len(target_points)
            observer_exposure_scores.append(exposure_score)

        # Average exposure across all observer positions
        avg_exposure = np.mean(observer_exposure_scores)

        logger.debug(f"Exposure {observer_zone} -> {target_zone}: {avg_exposure:.3f}")
        return float(avg_exposure)

    def calculate_pixel_exposure_map(self,
                                     observer_zone: str,
                                     target_zone: str,
                                     map_data: MapData) -> np.ndarray:
        """
        Calculate per-pixel exposure map using exhaustive search.

        For each pixel in the target zone, calculates what percentage of
        observer zone pixels can see it.

        Args:
            observer_zone: Name of the zone containing observers
            target_zone: Name of the zone being observed
            map_data: MapData object containing heightmap and zones

        Returns:
            2D array same size as map, with exposure value (0-1) for each pixel
        """
        logger.info(f"Calculating pixel-by-pixel exposure: {observer_zone} -> {target_zone}")

        height, width = map_data.heightmap.shape
        exposure_map = np.zeros((height, width), dtype=np.float32)

        # Get zone masks
        observer_mask = map_data.get_zone_mask(observer_zone)
        target_mask = map_data.get_zone_mask(target_zone)

        # Get all pixels in each zone
        observer_pixels = np.argwhere(observer_mask)  # [(y1, x1), (y2, x2), ...]
        target_pixels = np.argwhere(target_mask)

        total_observers = len(observer_pixels)
        total_targets = len(target_pixels)

        if total_observers == 0 or total_targets == 0:
            logger.warning(f"Empty zone found: observers={total_observers}, targets={total_targets}")
            return exposure_map

        logger.info(f"Full analysis: {total_observers:,} observers × {total_targets:,} targets = {total_observers * total_targets:,} checks")

        # For each target pixel, count how many observers can see it
        for idx, (tgt_y, tgt_x) in enumerate(target_pixels):
            if idx % 100 == 0:
                progress = (idx / total_targets) * 100
                logger.debug(f"Progress: {idx}/{total_targets} ({progress:.1f}%)")

            visible_count = 0

            # Check visibility from all observer pixels
            for obs_y, obs_x in observer_pixels:
                if self.check_line_of_sight(
                    (obs_x, obs_y),
                    (tgt_x, tgt_y),
                    map_data.heightmap
                ):
                    visible_count += 1

            # Exposure = fraction of observers that can see this pixel
            exposure_map[tgt_y, tgt_x] = visible_count / total_observers

        logger.info(f"Pixel exposure map complete: mean={exposure_map[target_mask].mean():.3f}")
        return exposure_map

    def analyze_map(self,
                   map_data: MapData,
                   analysis_pairs: List[Tuple[str, str]] = None,
                   num_samples: int = 100) -> Dict[str, float]:
        """
        Perform comprehensive visibility analysis on a map.

        Args:
            map_data: MapData object to analyze
            analysis_pairs: List of (observer_zone, target_zone) pairs to analyze.
                          If None, analyzes all zone combinations.
            num_samples: Number of sample points per zone

        Returns:
            Dictionary of analysis metrics including exposure ratios
        """
        logger.info(f"Starting visibility analysis with {num_samples} samples per zone")

        if analysis_pairs is None:
            # Create all possible zone pairs
            zone_names = list(map_data.zones.keys())
            analysis_pairs = [
                (obs, tgt) for obs in zone_names for tgt in zone_names if obs != tgt
            ]

        logger.debug(f"Analyzing {len(analysis_pairs)} zone pairs")

        metrics = {}

        # Calculate exposure for each pair
        for observer_zone, target_zone in analysis_pairs:
            exposure = self.calculate_zone_exposure(
                observer_zone,
                target_zone,
                map_data,
                num_samples,
                num_samples
            )

            metric_name = f"exposure_{observer_zone}_to_{target_zone}"
            metrics[metric_name] = exposure

        # Calculate aggregate metrics
        if 'team_a_zone' in map_data.zones and 'team_b_zone' in map_data.zones:
            # Team A's vulnerability (how much B can see A)
            team_a_exposure = metrics.get('exposure_team_b_zone_to_team_a_zone', 0.0)
            # Team B's vulnerability (how much A can see B)
            team_b_exposure = metrics.get('exposure_team_a_zone_to_team_b_zone', 0.0)

            metrics['team_a_exposure'] = team_a_exposure
            metrics['team_b_exposure'] = team_b_exposure

            # Maximum exposure (worst case vulnerability)
            metrics['max_exposure'] = max(team_a_exposure, team_b_exposure)

            # Exposure difference (balance metric)
            metrics['exposure_difference'] = abs(team_a_exposure - team_b_exposure)

            # Average exposure
            metrics['avg_exposure'] = (team_a_exposure + team_b_exposure) / 2.0

            logger.info(f"Analysis complete: Team A={team_a_exposure:.3f}, Team B={team_b_exposure:.3f}, Diff={metrics['exposure_difference']:.3f}")

        return metrics

    def analyze_map_full(self,
                        map_data: MapData,
                        analysis_pairs: List[Tuple[str, str]] = None) -> Dict[str, any]:
        """
        Perform pixel-by-pixel comprehensive visibility analysis on a map.

        This uses exhaustive search instead of sampling, calculating exposure
        for every pixel in each zone.

        Args:
            map_data: MapData object to analyze
            analysis_pairs: List of (observer_zone, target_zone) pairs to analyze.
                          If None, analyzes all zone combinations.

        Returns:
            Dictionary with metrics and per-pixel exposure maps
        """
        logger.info("Starting FULL pixel-by-pixel visibility analysis")

        if analysis_pairs is None:
            # Create all possible zone pairs
            zone_names = list(map_data.zones.keys())
            analysis_pairs = [
                (obs, tgt) for obs in zone_names for tgt in zone_names if obs != tgt
            ]

        logger.info(f"Analyzing {len(analysis_pairs)} zone pairs")

        metrics = {}
        exposure_maps = {}

        # Calculate exposure for each pair with full pixel maps
        for observer_zone, target_zone in analysis_pairs:
            exposure_map = self.calculate_pixel_exposure_map(
                observer_zone,
                target_zone,
                map_data
            )

            # Store the exposure map
            map_name = f"exposure_map_{observer_zone}_to_{target_zone}"
            exposure_maps[map_name] = exposure_map

            # Calculate average exposure across the target zone
            target_mask = map_data.get_zone_mask(target_zone)
            avg_exposure = exposure_map[target_mask].mean() if target_mask.any() else 0.0

            metric_name = f"exposure_{observer_zone}_to_{target_zone}"
            metrics[metric_name] = float(avg_exposure)

        # Calculate aggregate metrics
        if 'team_a_zone' in map_data.zones and 'team_b_zone' in map_data.zones:
            # Team A's vulnerability (how much B can see A)
            team_a_exposure = metrics.get('exposure_team_b_zone_to_team_a_zone', 0.0)
            # Team B's vulnerability (how much A can see B)
            team_b_exposure = metrics.get('exposure_team_a_zone_to_team_b_zone', 0.0)

            metrics['team_a_exposure'] = team_a_exposure
            metrics['team_b_exposure'] = team_b_exposure

            # Maximum exposure (worst case vulnerability)
            metrics['max_exposure'] = max(team_a_exposure, team_b_exposure)

            # Exposure difference (balance metric)
            metrics['exposure_difference'] = abs(team_a_exposure - team_b_exposure)

            # Average exposure
            metrics['avg_exposure'] = (team_a_exposure + team_b_exposure) / 2.0

            logger.info(f"Full analysis complete: Team A={team_a_exposure:.3f}, Team B={team_b_exposure:.3f}, Diff={metrics['exposure_difference']:.3f}")

        return {
            'metrics': metrics,
            'exposure_maps': exposure_maps
        }


class TerrainAnalyzer:
    """
    Analyzes terrain characteristics such as elevation variance and coverage.
    """

    @staticmethod
    def calculate_elevation_stats(heightmap: np.ndarray,
                                  mask: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate elevation statistics for a region.

        Args:
            heightmap: 2D elevation array
            mask: Optional boolean mask defining region of interest

        Returns:
            Dictionary containing mean, std, min, max elevation values
        """
        if mask is not None:
            heights = heightmap[mask]
        else:
            heights = heightmap.flatten()

        if len(heights) == 0:
            return {
                'mean_elevation': 0.0,
                'std_elevation': 0.0,
                'min_elevation': 0.0,
                'max_elevation': 0.0
            }

        return {
            'mean_elevation': float(np.mean(heights)),
            'std_elevation': float(np.std(heights)),
            'min_elevation': float(np.min(heights)),
            'max_elevation': float(np.max(heights))
        }

    @staticmethod
    def calculate_zone_terrain_stats(map_data: MapData) -> Dict[str, Dict[str, float]]:
        """
        Calculate terrain statistics for each zone in the map.

        Args:
            map_data: MapData object to analyze

        Returns:
            Nested dictionary with zone names as keys and stats dicts as values
        """
        zone_stats = {}

        for zone_name, zone in map_data.zones.items():
            mask = map_data.get_zone_mask(zone_name)
            stats = TerrainAnalyzer.calculate_elevation_stats(map_data.heightmap, mask)
            zone_stats[zone_name] = stats

        return zone_stats


def create_full_analysis(map_data: MapData,
                        num_samples: int = 100,
                        observer_height: float = 0.05,
                        use_full_analysis: bool = True) -> Dict[str, any]:
    """
    Convenience function to perform complete spatial analysis on a map.

    Args:
        map_data: MapData object to analyze
        num_samples: Number of sample points for visibility analysis (only if use_full_analysis=False)
        observer_height: Height offset for observers (normalized 0-1 scale)
        use_full_analysis: If True, uses pixel-by-pixel exhaustive analysis.
                          If False, uses sampling approach (faster).

    Returns:
        Dictionary containing all analysis results including per-pixel exposure maps
    """
    # Visibility analysis
    visibility_analyzer = VisibilityAnalyzer(observer_height=observer_height)

    if use_full_analysis:
        logger.info("Using FULL pixel-by-pixel analysis (exhaustive search)")
        result = visibility_analyzer.analyze_map_full(map_data)
        visibility_metrics = result['metrics']
        exposure_maps = result['exposure_maps']
    else:
        logger.info(f"Using sampling analysis ({num_samples} samples per zone)")
        visibility_metrics = visibility_analyzer.analyze_map(map_data, num_samples=num_samples)
        exposure_maps = {}

    # Terrain analysis
    terrain_stats = TerrainAnalyzer.calculate_zone_terrain_stats(map_data)

    # Combine results
    analysis_results = {
        'visibility_metrics': visibility_metrics,
        'terrain_stats': terrain_stats,
        'metadata': map_data.metadata.copy()
    }

    # Add exposure maps if available
    if exposure_maps:
        # Extract team-specific maps for easier access
        analysis_results['team_a_exposure_map'] = exposure_maps.get('exposure_map_team_b_zone_to_team_a_zone')
        analysis_results['team_b_exposure_map'] = exposure_maps.get('exposure_map_team_a_zone_to_team_b_zone')
        analysis_results['all_exposure_maps'] = exposure_maps

    return analysis_results

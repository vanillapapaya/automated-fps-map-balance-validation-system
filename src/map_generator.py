"""
Map Generator Module

Generates procedural terrain using Perlin noise with configurable team zones.
"""

import numpy as np
from noise import pnoise2
from typing import Tuple, Dict, Any
from .logger import get_logger

logger = get_logger(__name__)


class TeamZone:
    """
    Represents a team's zone on the map.

    Attributes:
        name: Unique identifier for the team zone
        bounds: Tuple of (x_min, x_max, y_min, y_max) defining zone boundaries
    """

    def __init__(self, name: str, bounds: Tuple[int, int, int, int]):
        """
        Initialize a team zone.

        Args:
            name: Unique identifier for the zone (e.g., "team_a_zone")
            bounds: Zone boundaries as (x_min, x_max, y_min, y_max)

        Raises:
            ValueError: If bounds are invalid
        """
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Invalid zone bounds: min values must be less than max values")

        self.name = name
        self.bounds = bounds

    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if a point is within this zone.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is within zone bounds
        """
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x < x_max and y_min <= y < y_max

    def get_sample_points(self, num_samples: int) -> np.ndarray:
        """
        Generate uniformly distributed sample points within the zone.

        Args:
            num_samples: Number of points to sample

        Returns:
            Array of shape (num_samples, 2) containing (x, y) coordinates
        """
        x_min, x_max, y_min, y_max = self.bounds

        # Generate grid-based samples for better coverage
        samples_per_dim = int(np.sqrt(num_samples))
        x_samples = np.linspace(x_min, x_max - 1, samples_per_dim, dtype=int)
        y_samples = np.linspace(y_min, y_max - 1, samples_per_dim, dtype=int)

        xx, yy = np.meshgrid(x_samples, y_samples)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Return exactly num_samples points
        return points[:num_samples]


class MapData:
    """
    Container for generated map data including heightmap and zone information.

    Attributes:
        heightmap: 2D array of terrain elevations (normalized 0-1)
        zones: Dictionary mapping zone names to TeamZone objects
        dimensions: Tuple of (width, height)
        metadata: Additional map generation parameters
    """

    def __init__(self, heightmap: np.ndarray, zones: Dict[str, TeamZone], metadata: Dict[str, Any]):
        """
        Initialize map data container.

        Args:
            heightmap: Terrain elevation array
            zones: Dictionary of team zones
            metadata: Generation parameters and additional info
        """
        self.heightmap = heightmap
        self.zones = zones
        self.dimensions = heightmap.shape
        self.metadata = metadata

    def get_zone_mask(self, zone_name: str) -> np.ndarray:
        """
        Create a boolean mask for a specific zone.

        Args:
            zone_name: Name of the zone

        Returns:
            Boolean array where True indicates points within the zone

        Raises:
            KeyError: If zone_name doesn't exist
        """
        if zone_name not in self.zones:
            raise KeyError(f"Zone '{zone_name}' not found in map data")

        zone = self.zones[zone_name]
        mask = np.zeros(self.dimensions, dtype=bool)
        x_min, x_max, y_min, y_max = zone.bounds
        mask[y_min:y_max, x_min:x_max] = True

        return mask


class MapGenerator:
    """
    Generates procedural terrain maps using Perlin noise.
    """

    def __init__(self,
                 width: int = 512,
                 height: int = 512,
                 octaves: int = 6,
                 persistence: float = 0.5,
                 lacunarity: float = 2.0,
                 scale: float = 100.0,
                 seed: int = None):
        """
        Initialize the map generator with terrain parameters.

        Args:
            width: Map width in pixels
            height: Map height in pixels
            octaves: Number of Perlin noise octaves (detail levels)
            persistence: How quickly amplitude decreases per octave (0-1)
            lacunarity: Frequency multiplier between octaves (typically 2.0)
            scale: Overall scale of the noise pattern
            seed: Random seed for reproducibility (None for random)
        """
        if width <= 0 or height <= 0:
            raise ValueError("Map dimensions must be positive")
        if octaves < 1:
            raise ValueError("Octaves must be at least 1")
        if not 0 < persistence < 1:
            raise ValueError("Persistence must be between 0 and 1")
        if lacunarity <= 0:
            raise ValueError("Lacunarity must be positive")

        self.width = width
        self.height = height
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.scale = scale
        self.seed = seed if seed is not None else np.random.randint(0, 10000)

        logger.info(f"Initialized MapGenerator: {width}x{height}, octaves={octaves}, seed={self.seed}")

    def generate_heightmap(self) -> np.ndarray:
        """
        Generate a Perlin noise-based heightmap.

        Returns:
            2D numpy array of normalized elevation values (0-1)
        """
        logger.debug(f"Generating heightmap with Perlin noise...")
        heightmap = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                # Generate Perlin noise value
                noise_val = pnoise2(
                    x / self.scale,
                    y / self.scale,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    repeatx=self.width,
                    repeaty=self.height,
                    base=self.seed
                )
                heightmap[y, x] = noise_val

        # Normalize to 0-1 range
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

        logger.debug(f"Heightmap generated: min={heightmap.min():.3f}, max={heightmap.max():.3f}, mean={heightmap.mean():.3f}")
        return heightmap

    def create_default_zones(self) -> Dict[str, TeamZone]:
        """
        Create default two-team zones on opposite sides of the map.

        Returns:
            Dictionary containing 'team_a_zone' and 'team_b_zone'

        실험을 위해 예시로 가로로 3등분하여 왼쪽 부분을 A, 오른쪽 부분을 B로 설정
        """
        # Divide map into three sections: team A (left), neutral (center), team B (right)
        section_width = self.width // 3

        zones = {
            'team_a_zone': TeamZone(
                'team_a_zone',
                (0, section_width, 0, self.height)
            ),
            'team_b_zone': TeamZone(
                'team_b_zone',
                (2 * section_width, self.width, 0, self.height)
            )
        }

        return zones

    def generate_map(self, zones: Dict[str, TeamZone] = None) -> MapData:
        """
        Generate a complete map with heightmap and zone definitions.

        Args:
            zones: Optional custom zone definitions. If None, uses default zones.

        Returns:
            MapData object containing heightmap, zones, and metadata
        """
        logger.info("Starting map generation...")

        # Generate terrain
        heightmap = self.generate_heightmap()

        # Use provided zones or create defaults
        if zones is None:
            logger.debug("Creating default team zones")
            zones = self.create_default_zones()
        else:
            logger.debug(f"Using custom zones: {list(zones.keys())}")

        # Validate zones fit within map bounds
        for zone_name, zone in zones.items():
            x_min, x_max, y_min, y_max = zone.bounds
            if not (0 <= x_min < x_max <= self.width and 0 <= y_min < y_max <= self.height):
                raise ValueError(f"Zone '{zone_name}' exceeds map boundaries")
            logger.debug(f"Zone '{zone_name}' bounds: ({x_min}, {x_max}, {y_min}, {y_max})")

        # Package metadata
        metadata = {
            'width': self.width,
            'height': self.height,
            'octaves': self.octaves,
            'persistence': self.persistence,
            'lacunarity': self.lacunarity,
            'scale': self.scale,
            'seed': self.seed
        }

        logger.info(f"Map generation complete: {self.width}x{self.height}, {len(zones)} zones")
        return MapData(heightmap, zones, metadata)


def create_custom_zones(width: int, height: int, zone_configs: list) -> Dict[str, TeamZone]:
    """
    Helper function to create custom zones from configuration.

    Args:
        width: Map width
        height: Map height
        zone_configs: List of dicts with 'name' and 'bounds' keys

    Returns:
        Dictionary of TeamZone objects

    Example:
        zone_configs = [
            {'name': 'team_a_zone', 'bounds': (0, 200, 0, 512)},
            {'name': 'team_b_zone', 'bounds': (312, 512, 0, 512)}
        ]
    """
    zones = {}

    for config in zone_configs:
        name = config['name']
        bounds = tuple(config['bounds'])

        # Validate bounds
        if len(bounds) != 4:
            raise ValueError(f"Zone '{name}' must have exactly 4 boundary values")

        zones[name] = TeamZone(name, bounds)

    return zones

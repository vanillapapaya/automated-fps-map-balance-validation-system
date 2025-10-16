"""
Visualization Module

Creates visual representations of maps, zones, and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from typing import Dict, Any, Optional
from .map_generator import MapData


class MapVisualizer:
    """
    Creates visualizations for maps and analysis results.
    """

    def __init__(self, dpi: int = 100):
        """
        Initialize the visualizer.

        Args:
            dpi: Resolution for saved images
        """
        self.dpi = dpi

    def save_heightmap(self, map_data: MapData, filepath: str):
        """
        Save heightmap as a grayscale image.

        Args:
            map_data: MapData object
            filepath: Output file path (PNG format)
        """
        # Normalize to 0-255 range
        heightmap_normalized = (map_data.heightmap * 255).astype(np.uint8)

        # Create and save image
        img = Image.fromarray(heightmap_normalized, mode='L')
        img.save(filepath)

    def visualize_zones(self,
                       map_data: MapData,
                       filepath: str,
                       show_grid: bool = False):
        """
        Create visualization with terrain and zone boundaries.

        Args:
            map_data: MapData object
            filepath: Output file path
            show_grid: Whether to show grid lines
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)

        # Display heightmap as background
        im = ax.imshow(map_data.heightmap, cmap='terrain', origin='upper')

        # Define colors for zones
        zone_colors = {
            'team_a_zone': (1.0, 0.0, 0.0, 0.3),  # Red with transparency
            'team_b_zone': (0.0, 0.0, 1.0, 0.3),  # Blue with transparency
        }

        # Draw zone boundaries
        for zone_name, zone in map_data.zones.items():
            x_min, x_max, y_min, y_max = zone.bounds

            # Get color or use default
            color = zone_colors.get(zone_name, (0.5, 0.5, 0.5, 0.3))

            # Create rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color[:3] + (1.0,),  # Solid edge
                facecolor=color,  # Transparent fill
                label=zone_name
            )
            ax.add_patch(rect)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Elevation')

        # Configure axes
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Map Terrain with Team Zones')

        if show_grid:
            ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def visualize_exposure_heatmap(self,
                                   map_data: MapData,
                                   analysis_results: Dict[str, Any],
                                   filepath: str):
        """
        Create heatmap showing exposure intensity across the map.

        Args:
            map_data: MapData object
            analysis_results: Results from spatial analysis
            filepath: Output file path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=self.dpi)

        # Left: Terrain with zones
        ax1.imshow(map_data.heightmap, cmap='terrain', origin='upper')
        ax1.set_title('Terrain Map')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        # Draw zones on left plot
        zone_colors = ['red', 'blue', 'green', 'yellow', 'purple']
        for idx, (zone_name, zone) in enumerate(map_data.zones.items()):
            x_min, x_max, y_min, y_max = zone.bounds
            color = zone_colors[idx % len(zone_colors)]

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                label=zone_name
            )
            ax1.add_patch(rect)

        ax1.legend()

        # Right: Zone exposure overlay
        exposure_overlay = np.zeros(map_data.dimensions)

        # Create zone masks with their exposure values
        for zone_name, zone in map_data.zones.items():
            # Find exposure metrics for this zone
            exposure_val = 0.0

            # Check for exposure metrics in analysis results
            visibility_metrics = analysis_results.get('visibility_metrics', {})

            # Look for metrics where this zone is the target
            for metric_name, value in visibility_metrics.items():
                if f"to_{zone_name}" in metric_name:
                    exposure_val = max(exposure_val, value)

            # Apply exposure value to zone area
            mask = map_data.get_zone_mask(zone_name)
            exposure_overlay[mask] = exposure_val

        # Create custom colormap (green to yellow to red)
        colors = ['green', 'yellow', 'orange', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('exposure', colors, N=n_bins)

        im = ax2.imshow(exposure_overlay, cmap=cmap, origin='upper', vmin=0, vmax=1)
        ax2.set_title('Zone Exposure Heatmap')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')

        plt.colorbar(im, ax=ax2, label='Exposure Ratio')

        # Add metrics text
        metrics = analysis_results.get('visibility_metrics', {})
        text_lines = []

        if 'team_a_exposure' in metrics and 'team_b_exposure' in metrics:
            text_lines.append(f"Team A Exposure: {metrics['team_a_exposure']:.3f}")
            text_lines.append(f"Team B Exposure: {metrics['team_b_exposure']:.3f}")

        if 'exposure_difference' in metrics:
            text_lines.append(f"Difference: {metrics['exposure_difference']:.3f}")

        if text_lines:
            ax2.text(0.02, 0.98, '\n'.join(text_lines),
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def create_analysis_report(self,
                              map_data: MapData,
                              analysis_results: Dict[str, Any],
                              validation_result: Any,
                              filepath: str):
        """
        Create comprehensive visual report with multiple subplots.

        Args:
            map_data: MapData object
            analysis_results: Results from spatial analysis
            validation_result: ValidationResult object
            filepath: Output file path
        """
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Heightmap
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(map_data.heightmap, cmap='terrain', origin='upper')
        ax1.set_title('Heightmap')
        ax1.axis('off')

        # 2. Zones overlay
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(map_data.heightmap, cmap='gray', origin='upper')

        zone_colors = {'team_a_zone': 'red', 'team_b_zone': 'blue'}
        for zone_name, zone in map_data.zones.items():
            x_min, x_max, y_min, y_max = zone.bounds
            color = zone_colors.get(zone_name, 'green')

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                label=zone_name
            )
            ax2.add_patch(rect)

        ax2.set_title('Team Zones')
        ax2.legend()
        ax2.axis('off')

        # 3. Exposure metrics bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = analysis_results.get('visibility_metrics', {})

        if 'team_a_exposure' in metrics and 'team_b_exposure' in metrics:
            teams = ['Team A', 'Team B']
            exposures = [metrics['team_a_exposure'], metrics['team_b_exposure']]
            colors = ['red', 'blue']

            bars = ax3.bar(teams, exposures, color=colors, alpha=0.7)
            ax3.axhline(y=0.4, color='orange', linestyle='--', label='Threshold (0.4)')
            ax3.set_ylabel('Exposure Ratio')
            ax3.set_title('Team Exposure Comparison')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)

        # 4. Validation results text
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        summary = validation_result.get_summary()

        # Create validation text
        validation_text = []
        validation_text.append("=" * 80)
        validation_text.append(f"VALIDATION RESULTS: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
        validation_text.append("=" * 80)
        validation_text.append(f"Total Rules: {summary['total_rules']}")
        validation_text.append(f"Passed: {summary['passed_rules']}")
        validation_text.append(f"Failed: {summary['failed_rules']}")
        validation_text.append("")

        if summary['failed_rules'] > 0:
            validation_text.append("Failed Rules:")
            for detail in summary['failed_rule_details']:
                validation_text.append(f"  - {detail['rule_id']}: {detail['message']}")
        else:
            validation_text.append("All validation rules passed!")

        validation_text.append("")
        validation_text.append("Key Metrics:")
        for key, value in summary['all_metrics'].items():
            if isinstance(value, (int, float)):
                validation_text.append(f"  {key}: {value:.4f}")

        # Display text
        text = '\n'.join(validation_text)
        ax4.text(0.05, 0.95, text,
                transform=ax4.transAxes,
                verticalalignment='top',
                fontfamily='monospace',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()


def save_analysis_json(analysis_results: Dict[str, Any],
                       validation_result: Any,
                       filepath: str):
    """
    Save analysis and validation results as JSON.

    Args:
        analysis_results: Spatial analysis results
        validation_result: ValidationResult object
        filepath: Output JSON file path
    """
    import json

    output_data = {
        'analysis_results': analysis_results,
        'validation_summary': validation_result.get_summary()
    }

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

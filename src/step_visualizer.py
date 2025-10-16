"""
Step-by-Step Visualization Module

Creates detailed visualizations for each stage of the map generation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import os

from .map_generator import MapData
from .logger import get_logger

logger = get_logger(__name__)


class StepByStepVisualizer:
    """
    Creates visualizations for each step of the map generation process.
    """

    def __init__(self, output_dir: str = "output/steps", dpi: int = 100):
        """
        Initialize the step visualizer.

        Args:
            output_dir: Directory to save step visualizations
            dpi: Resolution for saved images
        """
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"StepByStepVisualizer initialized: output_dir={output_dir}")

    def step1_raw_heightmap(self, map_data: MapData, step_id: str = "step1"):
        """
        Visualize Step 1: Raw heightmap from Perlin noise.

        Args:
            map_data: MapData object
            step_id: Identifier for this step
        """
        logger.info(f"Creating Step 1 visualization: Raw heightmap")

        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)

        # Display heightmap
        im = ax.imshow(map_data.heightmap, cmap='terrain', origin='upper')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Elevation')

        # Add statistics text
        stats_text = (
            f"Map Size: {map_data.dimensions[0]}x{map_data.dimensions[1]}\n"
            f"Min Elevation: {map_data.heightmap.min():.3f}\n"
            f"Max Elevation: {map_data.heightmap.max():.3f}\n"
            f"Mean Elevation: {map_data.heightmap.mean():.3f}\n"
            f"Std Elevation: {map_data.heightmap.std():.3f}\n\n"
            f"Perlin Noise Parameters:\n"
            f"  Octaves: {map_data.metadata['octaves']}\n"
            f"  Persistence: {map_data.metadata['persistence']}\n"
            f"  Lacunarity: {map_data.metadata['lacunarity']}\n"
            f"  Seed: {map_data.metadata['seed']}"
        )

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9,
                fontfamily='monospace')

        ax.set_title('Step 1: Raw Heightmap from Perlin Noise', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        filepath = os.path.join(self.output_dir, f"{step_id}_raw_heightmap.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Step 1 visualization saved: {filepath}")
        return filepath

    def step2_team_zones(self, map_data: MapData, step_id: str = "step2"):
        """
        Visualize Step 2: Team zones overlay on heightmap.

        Args:
            map_data: MapData object
            step_id: Identifier for this step
        """
        logger.info(f"Creating Step 2 visualization: Team zones")

        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)

        # Display heightmap
        ax.imshow(map_data.heightmap, cmap='gray', alpha=0.7, origin='upper')

        # Define colors for zones
        zone_colors = {
            'team_a_zone': (1.0, 0.0, 0.0, 0.3),  # Red
            'team_b_zone': (0.0, 0.0, 1.0, 0.3),  # Blue
        }

        zone_info = []

        # Draw zone boundaries
        for zone_name, zone in map_data.zones.items():
            x_min, x_max, y_min, y_max = zone.bounds
            color = zone_colors.get(zone_name, (0.5, 0.5, 0.5, 0.3))

            # Create rectangle
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=3,
                edgecolor=color[:3] + (1.0,),
                facecolor=color,
                label=zone_name
            )
            ax.add_patch(rect)

            # Add zone label
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            ax.text(center_x, center_y, zone_name.upper().replace('_', ' '),
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=14,
                   fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round', facecolor=color[:3], alpha=0.8))

            zone_info.append(f"{zone_name}: [{x_min}, {x_max}] x [{y_min}, {y_max}]")

        # Add zone info text
        zone_text = "Team Zones:\n" + "\n".join(zone_info)
        ax.text(0.02, 0.98, zone_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9,
                fontfamily='monospace')

        ax.set_title('Step 2: Team Zone Assignment', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend(loc='upper right')

        filepath = os.path.join(self.output_dir, f"{step_id}_team_zones.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Step 2 visualization saved: {filepath}")
        return filepath

    def step3_sample_points(self, map_data: MapData, num_samples: int = 50, step_id: str = "step3"):
        """
        Visualize Step 3: Sample points for visibility analysis.

        Args:
            map_data: MapData object
            num_samples: Number of sample points per zone
            step_id: Identifier for this step
        """
        logger.info(f"Creating Step 3 visualization: Sample points ({num_samples} per zone)")

        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)

        # Display heightmap
        ax.imshow(map_data.heightmap, cmap='terrain', alpha=0.6, origin='upper')

        # Draw zones
        zone_colors = {
            'team_a_zone': 'red',
            'team_b_zone': 'blue',
        }

        total_points = 0

        for zone_name, zone in map_data.zones.items():
            x_min, x_max, y_min, y_max = zone.bounds
            color = zone_colors.get(zone_name, 'gray')

            # Draw zone boundary
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            # Get and plot sample points
            sample_points = zone.get_sample_points(num_samples)
            ax.scatter(sample_points[:, 0], sample_points[:, 1],
                      c=color, s=30, alpha=0.7, marker='o',
                      label=f"{zone_name} ({len(sample_points)} points)")

            total_points += len(sample_points)

        # Add info text
        info_text = (
            f"Sample Points Generation:\n"
            f"Total Points: {total_points}\n"
            f"Points per Zone: {num_samples}\n\n"
            f"These points will be used for:\n"
            f"  - Observer positions (red zone)\n"
            f"  - Target positions (blue zone)\n"
            f"  - Line-of-sight calculations"
        )

        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9,
                fontfamily='monospace')

        ax.set_title('Step 3: Sample Points for Visibility Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend(loc='upper right')

        filepath = os.path.join(self.output_dir, f"{step_id}_sample_points.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Step 3 visualization saved: {filepath}")
        return filepath

    def step4_line_of_sight_demo(self, map_data: MapData, num_demo_lines: int = 10, step_id: str = "step4"):
        """
        Visualize Step 4: Line-of-sight calculation demonstration.

        Args:
            map_data: MapData object
            num_demo_lines: Number of example sightlines to show
            step_id: Identifier for this step
        """
        logger.info(f"Creating Step 4 visualization: Line-of-sight demo")

        from .spatial_analyzer import VisibilityAnalyzer

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=self.dpi)

        # Left: Top-down view with sightlines
        ax1.imshow(map_data.heightmap, cmap='terrain', alpha=0.6, origin='upper')

        # Get sample points
        team_a_zone = map_data.zones['team_a_zone']
        team_b_zone = map_data.zones['team_b_zone']

        observer_points = team_a_zone.get_sample_points(num_demo_lines)
        target_points = team_b_zone.get_sample_points(num_demo_lines)

        # Draw zones
        for zone_name, zone in map_data.zones.items():
            x_min, x_max, y_min, y_max = zone.bounds
            color = 'red' if zone_name == 'team_a_zone' else 'blue'
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
            ax1.add_patch(rect)

        # Initialize visibility analyzer
        analyzer = VisibilityAnalyzer(observer_height=0.05)

        visible_count = 0
        blocked_count = 0

        # Draw sightlines
        for i in range(min(num_demo_lines, len(observer_points), len(target_points))):
            obs = tuple(observer_points[i])
            tgt = tuple(target_points[i])

            # Check visibility
            is_visible = analyzer.check_line_of_sight(obs, tgt, map_data.heightmap)

            # Draw line
            color = 'green' if is_visible else 'red'
            alpha = 0.6 if is_visible else 0.4
            linestyle = '-' if is_visible else ':'

            ax1.plot([obs[0], tgt[0]], [obs[1], tgt[1]],
                    color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5)

            # Draw points
            ax1.scatter(*obs, c='red', s=50, zorder=5, edgecolors='white', linewidth=1)
            ax1.scatter(*tgt, c='blue', s=50, zorder=5, edgecolors='white', linewidth=1)

            if is_visible:
                visible_count += 1
            else:
                blocked_count += 1

        ax1.set_title('Line-of-Sight Calculations\n(Green=Visible, Red=Blocked)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        # Right: 3D elevation profile
        ax2.remove()
        ax2 = fig.add_subplot(122, projection='3d')

        # Create 3D surface
        x = np.arange(map_data.heightmap.shape[1])
        y = np.arange(map_data.heightmap.shape[0])
        X, Y = np.meshgrid(x, y)

        # Downsample for performance
        stride = max(1, map_data.heightmap.shape[0] // 100)
        surf = ax2.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                               map_data.heightmap[::stride, ::stride],
                               cmap='terrain', alpha=0.7, edgecolor='none')

        ax2.set_title('3D Elevation Profile', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Elevation')

        # Add info text
        info_text = (
            f"Visibility Analysis:\n"
            f"Sample Lines: {num_demo_lines}\n"
            f"Visible: {visible_count} ({visible_count/num_demo_lines*100:.1f}%)\n"
            f"Blocked: {blocked_count} ({blocked_count/num_demo_lines*100:.1f}%)\n\n"
            f"Algorithm:\n"
            f"  1. Cast ray from observer to target\n"
            f"  2. Check terrain height along path\n"
            f"  3. If terrain blocks view → Blocked\n"
            f"  4. Otherwise → Visible"
        )

        fig.text(0.02, 0.5, info_text,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9,
                fontfamily='monospace')

        filepath = os.path.join(self.output_dir, f"{step_id}_line_of_sight.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Step 4 visualization saved: {filepath}")
        return filepath

    def step5_exposure_calculation(self, map_data: MapData, analysis_results: Dict,
                                   step_id: str = "step5"):
        """
        Visualize Step 5: Exposure calculation results with per-pixel exposure maps.

        Args:
            map_data: MapData object
            analysis_results: Analysis results from spatial analyzer
            step_id: Identifier for this step
        """
        logger.info(f"Creating Step 5 visualization: Exposure calculation")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14), dpi=self.dpi)

        metrics = analysis_results.get('visibility_metrics', {})

        # Check if we have per-pixel exposure maps (from full analysis)
        team_a_exposure_map = analysis_results.get('team_a_exposure_map')
        team_b_exposure_map = analysis_results.get('team_b_exposure_map')

        team_a_exposure = metrics.get('team_a_exposure', 0.0)
        team_b_exposure = metrics.get('team_b_exposure', 0.0)

        # 1. Team A Zone Exposure
        if team_a_exposure_map is not None:
            # Use actual per-pixel exposure map
            exposure_map_a = team_a_exposure_map
            logger.info("Using per-pixel exposure map for Team A")
        else:
            # Fallback: uniform exposure for entire zone
            team_a_mask = map_data.get_zone_mask('team_a_zone')
            exposure_map_a = np.zeros(map_data.dimensions)
            exposure_map_a[team_a_mask] = team_a_exposure
            logger.info("Using uniform exposure for Team A (sampling mode)")

        im1 = ax1.imshow(exposure_map_a, cmap='Reds', origin='upper', vmin=0, vmax=1)
        ax1.set_title(f'Team A Exposure: {team_a_exposure:.3f}\n(How vulnerable A is to B)', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Exposure Ratio')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        # Add pixel count info if using full analysis
        if team_a_exposure_map is not None:
            team_a_mask = map_data.get_zone_mask('team_a_zone')
            exposed_pixels = np.sum(team_a_exposure_map[team_a_mask] > 0.5)
            total_pixels = np.sum(team_a_mask)
            ax1.text(0.02, 0.98, f'Highly exposed:\n{exposed_pixels}/{total_pixels} pixels',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)

        # 2. Team B Zone Exposure
        if team_b_exposure_map is not None:
            # Use actual per-pixel exposure map
            exposure_map_b = team_b_exposure_map
            logger.info("Using per-pixel exposure map for Team B")
        else:
            # Fallback: uniform exposure for entire zone
            team_b_mask = map_data.get_zone_mask('team_b_zone')
            exposure_map_b = np.zeros(map_data.dimensions)
            exposure_map_b[team_b_mask] = team_b_exposure
            logger.info("Using uniform exposure for Team B (sampling mode)")

        im2 = ax2.imshow(exposure_map_b, cmap='Blues', origin='upper', vmin=0, vmax=1)
        ax2.set_title(f'Team B Exposure: {team_b_exposure:.3f}\n(How vulnerable B is to A)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Exposure Ratio')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')

        # Add pixel count info if using full analysis
        if team_b_exposure_map is not None:
            team_b_mask = map_data.get_zone_mask('team_b_zone')
            exposed_pixels = np.sum(team_b_exposure_map[team_b_mask] > 0.5)
            total_pixels = np.sum(team_b_mask)
            ax2.text(0.02, 0.98, f'Highly exposed:\n{exposed_pixels}/{total_pixels} pixels',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)

        # 3. Exposure Difference Map
        exposure_diff_map = np.abs(exposure_map_a - exposure_map_b)
        im3 = ax3.imshow(exposure_diff_map, cmap='RdYlGn_r', origin='upper', vmin=0, vmax=0.5)
        ax3.set_title('Exposure Difference Map\n(Red=Imbalanced, Green=Balanced)', fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='Absolute Difference')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')

        # 4. Metrics Summary
        ax4.axis('off')

        exposure_diff = metrics.get('exposure_difference', 0.0)
        avg_exposure = metrics.get('avg_exposure', 0.0)
        max_exposure = metrics.get('max_exposure', 0.0)

        # Create bar chart - use numeric positions to avoid categorical data issues
        ax4_inset = ax4.inset_axes([0.1, 0.5, 0.8, 0.4])
        teams_labels = ['Team A', 'Team B']
        x_positions = [0, 1]
        exposures = [team_a_exposure, team_b_exposure]
        colors = ['red', 'blue']

        bars = ax4_inset.bar(x_positions, exposures, color=colors, alpha=0.7, width=0.6)
        ax4_inset.axhline(y=0.4, color='orange', linestyle='--', label='Max Threshold')
        ax4_inset.set_ylabel('Exposure Ratio')
        ax4_inset.set_ylim(0, 1)
        ax4_inset.set_xticks(x_positions)
        ax4_inset.set_xticklabels(teams_labels)
        ax4_inset.set_title('Exposure Comparison')
        ax4_inset.legend()
        ax4_inset.grid(axis='y', alpha=0.3)

        # Add metrics text
        analysis_mode = "FULL (Pixel-by-Pixel)" if (team_a_exposure_map is not None) else "SAMPLING"
        metrics_text = (
            f"EXPOSURE ANALYSIS RESULTS\n"
            f"{'='*50}\n"
            f"Analysis Mode: {analysis_mode}\n\n"
            f"Team A Exposure:     {team_a_exposure:.4f}\n"
            f"Team B Exposure:     {team_b_exposure:.4f}\n"
            f"Exposure Difference: {exposure_diff:.4f}\n"
            f"Average Exposure:    {avg_exposure:.4f}\n"
            f"Maximum Exposure:    {max_exposure:.4f}\n\n"
            f"Balance Score: {'BALANCED' if exposure_diff < 0.15 else 'UNBALANCED'}\n"
            f"  (Difference < 0.15 = Balanced)"
        )

        ax4.text(0.1, 0.35, metrics_text,
                fontsize=10,
                fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        filepath = os.path.join(self.output_dir, f"{step_id}_exposure_calculation.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Step 5 visualization saved: {filepath}")
        return filepath

    def step6_validation_results(self, validation_result, step_id: str = "step6"):
        """
        Create Step 6: Validation results as text report.

        Args:
            validation_result: ValidationResult object
            step_id: Identifier for this step
        """
        logger.info(f"Creating Step 6 text report: Validation results")

        summary = validation_result.get_summary()

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STEP 6: VALIDATION RESULTS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall status
        if summary['overall_passed']:
            report_lines.append("OVERALL STATUS: ✓ PASSED")
            report_lines.append("")
            report_lines.append("This map meets all balance criteria and is APPROVED for use.")
        else:
            report_lines.append("OVERALL STATUS: ✗ FAILED")
            report_lines.append("")
            report_lines.append("This map fails to meet balance requirements and is REJECTED.")

        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Rules:   {summary['total_rules']}")
        report_lines.append(f"Passed Rules:  {summary['passed_rules']}")
        report_lines.append(f"Failed Rules:  {summary['failed_rules']}")
        report_lines.append(f"Pass Rate:     {summary['passed_rules']/summary['total_rules']*100:.1f}%")
        report_lines.append("")

        # Rule details
        report_lines.append("-" * 80)
        report_lines.append("RULE DETAILS")
        report_lines.append("-" * 80)
        report_lines.append("")

        for rule_id, (passed, value, msg) in validation_result.rule_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report_lines.append(f"[{status}] {msg}")
            report_lines.append(f"       Calculated Value: {value:.4f}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Write to file
        filepath = os.path.join(self.output_dir, f"{step_id}_validation_results.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Step 6 text report saved: {filepath}")
        return filepath

    def create_all_steps(self, map_data: MapData, analysis_results: Dict,
                        validation_result, num_samples: int = 50,
                        step_prefix: str = "step"):
        """
        Create all step visualizations at once.

        Args:
            map_data: MapData object
            analysis_results: Analysis results from spatial analyzer
            validation_result: ValidationResult object
            num_samples: Number of samples for step 3
            step_prefix: Prefix for step filenames

        Returns:
            List of created file paths
        """
        logger.info("Creating all step-by-step visualizations...")

        filepaths = []

        filepaths.append(self.step1_raw_heightmap(map_data, f"{step_prefix}1"))
        filepaths.append(self.step2_team_zones(map_data, f"{step_prefix}2"))
        filepaths.append(self.step3_sample_points(map_data, num_samples, f"{step_prefix}3"))
        filepaths.append(self.step4_line_of_sight_demo(map_data, 10, f"{step_prefix}4"))
        filepaths.append(self.step5_exposure_calculation(map_data, analysis_results, f"{step_prefix}5"))
        filepaths.append(self.step6_validation_results(validation_result, f"{step_prefix}6"))

        logger.info(f"All {len(filepaths)} step visualizations created")

        return filepaths

"""
Validation Engine Module

Defines and evaluates rules to validate map balance and quality.
"""

import json
from typing import Dict, List, Any, Optional
from enum import Enum
from .logger import get_logger

logger = get_logger(__name__)


class ComparisonOperator(Enum):
    """Comparison operators for validation rules."""
    LESS_THAN = "less_than"
    LESS_EQUAL = "less_equal"
    GREATER_THAN = "greater_than"
    GREATER_EQUAL = "greater_equal"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    BETWEEN = "between"


class ValidationRule:
    """
    Represents a single validation rule for map quality checking.

    Attributes:
        rule_id: Unique identifier for the rule
        metric_name: Name of the metric to evaluate
        operator: Comparison operator to use
        threshold: Value(s) to compare against
        description: Human-readable description of the rule
        zone: Optional zone name if rule applies to specific zone
    """

    def __init__(self,
                 rule_id: str,
                 metric_name: str,
                 operator: str,
                 threshold: Any,
                 description: str = "",
                 zone: Optional[str] = None):
        """
        Initialize a validation rule.

        Args:
            rule_id: Unique rule identifier
            metric_name: Name of metric in analysis results to check
            operator: Comparison operator (less_than, greater_than, between, etc.)
            threshold: Threshold value or [min, max] for 'between' operator
            description: Description of what this rule checks
            zone: Optional zone name for zone-specific rules

        Raises:
            ValueError: If operator is invalid or threshold format is wrong
        """
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.description = description
        self.zone = zone

        # Validate and set operator
        try:
            self.operator = ComparisonOperator(operator)
        except ValueError:
            valid_ops = [op.value for op in ComparisonOperator]
            raise ValueError(f"Invalid operator '{operator}'. Must be one of: {valid_ops}")

        # Validate threshold format
        if self.operator == ComparisonOperator.BETWEEN:
            if not isinstance(threshold, (list, tuple)) or len(threshold) != 2:
                raise ValueError("'between' operator requires threshold as [min, max]")
            if threshold[0] >= threshold[1]:
                raise ValueError("'between' threshold: min must be less than max")
        else:
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Operator '{operator}' requires numeric threshold")

        self.threshold = threshold

    def evaluate(self, metric_value: float) -> bool:
        """
        Evaluate the rule against a metric value.

        Args:
            metric_value: The value to check

        Returns:
            True if the value passes the rule, False otherwise
        """
        if self.operator == ComparisonOperator.LESS_THAN:
            return metric_value < self.threshold

        elif self.operator == ComparisonOperator.LESS_EQUAL:
            return metric_value <= self.threshold

        elif self.operator == ComparisonOperator.GREATER_THAN:
            return metric_value > self.threshold

        elif self.operator == ComparisonOperator.GREATER_EQUAL:
            return metric_value >= self.threshold

        elif self.operator == ComparisonOperator.EQUAL:
            return abs(metric_value - self.threshold) < 1e-9

        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return abs(metric_value - self.threshold) >= 1e-9

        elif self.operator == ComparisonOperator.BETWEEN:
            return self.threshold[0] <= metric_value <= self.threshold[1]

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary format."""
        return {
            'rule_id': self.rule_id,
            'metric_name': self.metric_name,
            'operator': self.operator.value,
            'threshold': self.threshold,
            'description': self.description,
            'zone': self.zone
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationRule':
        """Create a ValidationRule from dictionary data."""
        return cls(
            rule_id=data['rule_id'],
            metric_name=data['metric_name'],
            operator=data['operator'],
            threshold=data['threshold'],
            description=data.get('description', ''),
            zone=data.get('zone')
        )


class ValidationResult:
    """
    Container for validation results.

    Attributes:
        passed: Whether all rules passed
        rule_results: Dictionary mapping rule_id to (passed, metric_value, message)
        metrics: All metrics that were evaluated
    """

    def __init__(self):
        self.passed = True
        self.rule_results: Dict[str, tuple] = {}
        self.metrics: Dict[str, float] = {}

    def add_rule_result(self,
                       rule_id: str,
                       passed: bool,
                       metric_value: Optional[float],
                       message: str):
        """
        Add a rule evaluation result.

        Args:
            rule_id: ID of the rule
            passed: Whether the rule passed
            metric_value: The metric value that was checked
            message: Description of the result
        """
        self.rule_results[rule_id] = (passed, metric_value, message)
        if not passed:
            self.passed = False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results.

        Returns:
            Dictionary with overall status and detailed results
        """
        passed_count = sum(1 for p, _, _ in self.rule_results.values() if p)
        failed_count = len(self.rule_results) - passed_count

        failed_rules = [
            {
                'rule_id': rule_id,
                'metric_value': value,
                'message': msg
            }
            for rule_id, (passed, value, msg) in self.rule_results.items()
            if not passed
        ]

        return {
            'overall_passed': self.passed,
            'total_rules': len(self.rule_results),
            'passed_rules': passed_count,
            'failed_rules': failed_count,
            'failed_rule_details': failed_rules,
            'all_metrics': self.metrics
        }


class ValidationEngine:
    """
    Evaluates maps against a set of validation rules.
    """

    def __init__(self, rules: List[ValidationRule] = None):
        """
        Initialize the validation engine.

        Args:
            rules: List of ValidationRule objects. If None, starts with empty rule set.
        """
        self.rules = rules if rules is not None else []

    def add_rule(self, rule: ValidationRule):
        """Add a validation rule to the engine."""
        # Check for duplicate rule IDs
        if any(r.rule_id == rule.rule_id for r in self.rules):
            raise ValueError(f"Rule with ID '{rule.rule_id}' already exists")
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule by ID.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was found and removed, False otherwise
        """
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < initial_count

    def validate(self, analysis_results: Dict[str, Any]) -> ValidationResult:
        """
        Validate analysis results against all rules.

        Args:
            analysis_results: Dictionary containing analysis metrics

        Returns:
            ValidationResult object with detailed results
        """
        logger.info(f"Starting validation with {len(self.rules)} rules")
        result = ValidationResult()

        # Extract metrics from analysis results
        if 'visibility_metrics' in analysis_results:
            result.metrics.update(analysis_results['visibility_metrics'])

        # Add terrain stats if present
        if 'terrain_stats' in analysis_results:
            for zone_name, stats in analysis_results['terrain_stats'].items():
                for stat_name, value in stats.items():
                    result.metrics[f"{zone_name}_{stat_name}"] = value

        logger.debug(f"Evaluating {len(result.metrics)} metrics")

        # Evaluate each rule
        for rule in self.rules:
            # Construct full metric name if zone is specified
            if rule.zone:
                metric_key = f"{rule.zone}_{rule.metric_name}"
            else:
                metric_key = rule.metric_name

            # Check if metric exists
            if metric_key not in result.metrics:
                logger.warning(f"Metric '{metric_key}' not found for rule '{rule.rule_id}'")
                result.add_rule_result(
                    rule.rule_id,
                    False,
                    None,
                    f"Metric '{metric_key}' not found in analysis results"
                )
                continue

            # Get metric value and evaluate
            metric_value = result.metrics[metric_key]
            passed = rule.evaluate(metric_value)

            # Create result message
            if passed:
                message = f"✓ {rule.description or rule.rule_id}: {metric_value:.4f}"
                logger.debug(f"Rule '{rule.rule_id}' PASSED: {metric_value:.4f}")
            else:
                message = (f"✗ {rule.description or rule.rule_id}: "
                          f"{metric_value:.4f} (threshold: {rule.threshold})")
                logger.debug(f"Rule '{rule.rule_id}' FAILED: {metric_value:.4f} (threshold: {rule.threshold})")

            result.add_rule_result(rule.rule_id, passed, metric_value, message)

        summary = result.get_summary()
        logger.info(f"Validation complete: {'PASSED' if summary['overall_passed'] else 'FAILED'} "
                   f"({summary['passed_rules']}/{summary['total_rules']} rules passed)")

        return result

    def load_rules_from_file(self, filepath: str):
        """
        Load validation rules from a JSON configuration file.

        Args:
            filepath: Path to JSON rules file

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            ValueError: If rule format is invalid
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        rules_data = data.get('rules', [])

        for rule_dict in rules_data:
            rule = ValidationRule.from_dict(rule_dict)
            self.add_rule(rule)

    def save_rules_to_file(self, filepath: str):
        """
        Save current rules to a JSON configuration file.

        Args:
            filepath: Path where to save the rules
        """
        rules_data = [rule.to_dict() for rule in self.rules]

        data = {
            'version': '1.0',
            'rules': rules_data
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def create_default_rules() -> List[ValidationRule]:
    """
    Create a set of default validation rules for balanced maps.

    Returns:
        List of ValidationRule objects
    """
    rules = [
        ValidationRule(
            rule_id='max_exposure_limit',
            metric_name='max_exposure',
            operator='less_than',
            threshold=0.4,
            description='Maximum sight exposure must be below 40%'
        ),
        ValidationRule(
            rule_id='exposure_balance',
            metric_name='exposure_difference',
            operator='less_than',
            threshold=0.15,
            description='Exposure difference between teams must be below 15%'
        ),
        ValidationRule(
            rule_id='minimum_exposure',
            metric_name='avg_exposure',
            operator='greater_than',
            threshold=0.1,
            description='Average exposure must exceed 10% to avoid stalemate'
        ),
        ValidationRule(
            rule_id='team_a_exposure_range',
            metric_name='team_a_exposure',
            operator='between',
            threshold=[0.1, 0.5],
            description='Team A exposure must be between 10% and 50%'
        ),
        ValidationRule(
            rule_id='team_b_exposure_range',
            metric_name='team_b_exposure',
            operator='between',
            threshold=[0.1, 0.5],
            description='Team B exposure must be between 10% and 50%'
        )
    ]

    return rules

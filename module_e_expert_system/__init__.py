"""Module E: Expert System (Maintenance Advisor)"""
from .engine import MaintenanceExpertEngine, DiagnosticRule, FiredRuleRecord
from .knowledge_acquisition import add_diagnostic_rule, interactive_rule_acquisition

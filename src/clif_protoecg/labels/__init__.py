"""Label system: registry, definitions, and batch evaluator."""

from clif_protoecg.labels.base import LabelDefinition, label, get_registry
from clif_protoecg.labels.evaluator import LabelEvaluator

__all__ = ["LabelDefinition", "label", "get_registry", "LabelEvaluator"]

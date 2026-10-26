from tabulate import tabulate
import mlx.core as mx
from mlx.utils import tree_flatten
from typing import List, Dict, Any

"""Hold Classes and functions to print out MLX model info"""


class LayerInfo:
    """Store layer/parameter info"""

    def __init__(self, name: str, shape: tuple, num_params: int, depth: int = 0):
        self.name = name
        self.shape = shape
        self.num_params = num_params
        self.depth = depth


class MLXModelStatistics:  #todo convert string into Python Tabulate (https://www.datacamp.com/tutorial/python-tabulate)
    """Container to get MLX model statistics and tree architecture"""

    def __init__(self, model, input_shape=None):
        self.model = model
        self.input_shape = input_shape
        self.layers: List[LayerInfo] = []
        self.total_params = 0
        self.hierarchy = {}
        self._analyze_model()

    @staticmethod
    def _get_param_name(name: str):
        parts = name.split('.')
        return parts

    def _analyze_model(self):
        params = tree_flatten(self.model.parameters())

        # build hierarchy
        for name, param in params:
            parts = self._get_param_name(name)
            self.total_params += param.size

            # tree parts
            current = self.hierarchy
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {
                        '_children': {},
                        '_param': None,
                        '_shape': None,
                        '_count': 0
                    }

                # if last tree piece, store info
                if i == len(parts) - 1:
                    current[part]['_param'] = param
                    current[part]['_shape'] = param.shape
                    current[part]['_count'] = param.size

                current = current[part]['_children']

    def _count_params(self, node: Dict) -> int:
        """Count params in a node"""
        total = node.get('_count', 0)
        for child in node.get('_children', {}).values():
            total += self._count_params(child)
        return total

    def _format_tree(self, node: Dict, name: str = "", prefix: str = "", is_last: bool = True,
                     is_root: bool = True) -> List[str]:
        """Format tree structure"""
        lines = []

        if not is_root:
            # Set tree connector
            connector = "└─" if is_last else "├─"

            # Count params
            param_count = self._count_params(node)

            # Format param count
            if param_count > 0:
                param_str = f"{param_count:,}"
            else:
                param_str = "--"

            # If leaf, format shape
            if node.get('_param') is not None:
                shape_str = str(node['_shape'])
            else:
                shape_str = "--"

            # Build connector
            line = f"{prefix}{connector}({name})"
            line = f"{line:<50} {shape_str:<25} {param_str:>15}"
            lines.append(line)

        # get prefix
        children = node.get('_children', {})
        if children:
            child_items = list(children.items())
            for i, (child_name, child_node) in enumerate(child_items):
                is_last_child = (i == len(child_items) - 1)

                if is_root:
                    child_prefix = ""
                else:
                    extension = "    " if is_last else "│   "
                    child_prefix = prefix + extension

                lines.extend(
                    self._format_tree(
                        child_node,
                        child_name,
                        child_prefix,
                        is_last_child,
                        is_root=False
                    )
                )

        return lines

    def __str__(self) -> str:
        """Return formatted string with model structure"""
        lines = []

        # Header
        model_name = self.model.__class__.__name__
        lines.append(f"{model_name:<50} {'Shape':<25} {'Params':>15}")
        lines.append("=" * 90)

        # Tree
        for name, node in self.hierarchy.items():
            lines.extend(self._format_tree(node, name, "", True, is_root=False))

        # Footer
        lines.append("=" * 90)
        lines.append(f"{'Total params':<50} {'':<25} {self.total_params:>15,}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MLXModelStatistics(total_params={self.total_params:,})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()

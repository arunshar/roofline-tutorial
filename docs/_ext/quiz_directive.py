"""Define the {quiz} directive, which shows clickable questions on Markdown pages.

The directive body is a YAML list in the same shape that tutorial_helpers.quiz() takes:

    ```{quiz}
    - q: Does a grid index help a query that covers the whole map?
      options: [Yes, No]
      answer: 1
      explain: The query touches every cell, so the index cannot skip anything.
    ```
"""

import sys
from pathlib import Path

import yaml
from docutils import nodes
from sphinx.util.docutils import SphinxDirective

LABS = Path(__file__).resolve().parents[2] / "labs"


class Quiz(SphinxDirective):
    has_content = True

    def run(self):
        if str(LABS) not in sys.path:
            sys.path.insert(0, str(LABS))
        from tutorial_helpers import quiz_html

        try:
            questions = yaml.safe_load("\n".join(self.content))
            markup = quiz_html(questions)
        except (yaml.YAMLError, ValueError) as err:
            raise self.error(f"The quiz block is not valid. {err}")
        return [nodes.raw("", markup, format="html")]


def setup(app):
    app.add_directive("quiz", Quiz)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}

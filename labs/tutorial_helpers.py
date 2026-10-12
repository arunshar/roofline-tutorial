"""Helpers that every lecture notebook imports.

quiz(questions) shows clickable multiple-choice questions in Jupyter, JupyterLab or Colab.
quiz_html(questions) returns the same widget as an HTML string. The {quiz} directive in
docs/_ext uses it, so a Markdown page and a notebook show identical quizzes.

Each question is a dict like this one:
    {"q": "Question text?", "options": ["A", "B"], "answer": 0, "explain": "Why A."}
The "answer" value counts the options from 0, and every string may hold HTML.
Put topic-specific helpers in the lecture notebooks, where readers can see the formulas.
"""

import hashlib
import json

_STYLE = """<style>
.tq-set{margin:.8em 0}
.tq{border:1px solid rgba(110,118,129,.4);border-radius:8px;padding:.55em 1em;margin:.75em 0}
.tq-q{margin:.2em 0 .5em;font-weight:600}
.tq-opt{display:block;box-sizing:border-box;width:100%;margin:.3em 0;padding:.4em .7em;
  text-align:left;font:inherit;color:inherit;cursor:pointer;
  background:none;border:1px solid rgba(110,118,129,.55);border-radius:5px}
.tq-opt:hover:enabled{background:rgba(110,118,129,.12)}
.tq-opt.tq-right{background:rgba(26,127,55,.16);border-color:#1a7f37}
.tq-opt.tq-wrong{background:rgba(207,34,46,.14);border-color:#cf222e}
.tq-fb{margin:.5em 0 .1em;padding:.3em .7em;border-left:3px solid #2b6cb0}
.tq-score{margin:.4em 0 0;font-size:92%;opacity:.85}
</style>"""

# This script runs from each button's onclick attribute, so the page needs no separate
# <script> block. It uses only single quotes because it sits inside a double-quoted attribute.
_ONCLICK = (
    "(function(b){var q=b.closest('.tq'),s=b.closest('.tq-set');"
    "var ok=b.getAttribute('data-ok')==='1';"
    "q.querySelectorAll('.tq-opt').forEach(function(x){x.classList.remove('tq-wrong');});"
    "if(!q.hasAttribute('data-tried')){q.setAttribute('data-tried','1');"
    "if(ok){s.setAttribute('data-score',(+s.getAttribute('data-score'))+1);}}"
    "b.classList.add(ok?'tq-right':'tq-wrong');"
    "var fb=q.querySelector('.tq-fb');fb.hidden=false;"
    "fb.innerHTML=ok?q.querySelector('template').innerHTML:'That answer is wrong. Try another option.';"
    "if(ok){q.querySelectorAll('.tq-opt').forEach(function(x){x.disabled=true;});}"
    "var tried=s.querySelectorAll('.tq[data-tried]').length,n=s.querySelectorAll('.tq').length;"
    "s.querySelector('.tq-score').textContent='Right on the first try: '"
    "+s.getAttribute('data-score')+' of '+tried+' answered, '+n+(n===1?' question':' questions')+' in all.';"
    "})(this)"
)


def _check(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError("quiz() needs a non-empty list of question dicts")
    for i, q in enumerate(questions, 1):
        missing = {"q", "options", "answer"} - set(q)
        if missing:
            raise ValueError(f"question {i} is missing {sorted(missing)}")
        if len(q["options"]) < 2:
            raise ValueError(f"question {i} needs at least two options")
        if not 0 <= int(q["answer"]) < len(q["options"]):
            raise ValueError(f"question {i} has answer {q['answer']}, which is not an option index")


def quiz_html(questions):
    """Return the quiz as one self-contained HTML string."""
    _check(questions)
    tag = hashlib.sha1(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:8]
    html = [_STYLE, f'<div class="tq-set" id="tq-{tag}" data-score="0">']
    for i, q in enumerate(questions, 1):
        html.append('<div class="tq">')
        html.append(f'<p class="tq-q">{i}. {q["q"]}</p>')
        for j, option in enumerate(q["options"]):
            ok = "1" if j == int(q["answer"]) else "0"
            html.append(f'<button type="button" class="tq-opt" data-ok="{ok}" '
                        f'onclick="{_ONCLICK}">{option}</button>')
        html.append(f'<template><b>That is right.</b> {q.get("explain", "")}</template>')
        html.append('<p class="tq-fb" hidden></p>')
        html.append("</div>")
    html.append('<p class="tq-score"></p>')
    html.append("</div>")
    return "\n".join(html)


def quiz(questions):
    """Show the quiz in a notebook."""
    from IPython.display import HTML, display
    display(HTML(quiz_html(questions)))

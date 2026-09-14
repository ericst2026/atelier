# Give it a calculator and teach it to reach for one

`project/tools.py` — `TOOLS`, a dict of name to callable, and `render(name, args, result)`
deciding how a call and its result appear in the context. `project/train.py`
generates traces and trains.

The grader runs your model through its own tool loop on unseen questions and scores
accuracy, the share of calls whose arguments parse, and how often the returned
result appears in the final answer.

What usually goes wrong, in order: the model calls a tool for everything, including
questions no tool can help with — fix it with demonstrations that call nothing. It
writes calls the parser cannot read — make the syntax narrower and the
demonstrations more uniform. It calls correctly and then ignores the result — put
the answer immediately after the result in your traces, with nothing between.

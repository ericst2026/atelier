# Reach further without forgetting how to be short

`project/extend.py` defines `extend(model, target_length)` — modify the model in
place and train if you want — and saves `outputs/model.pt`.

The grader plants a fact at ten depths in 2048-token documents and asks for it, then
checks the model is still as good as it was on short inputs. Both halves are scored,
because the cheap way to win the first is to wreck the second.

What works: position interpolation with a scale equal to the extension factor, a
few hundred steps of continuation on genuinely long documents, and putting some
needle-shaped documents in that continuation data so the model learns that distant
text can matter. What does not: raising the RoPE base alone by a large factor, or
training on short documents concatenated to look long.

# dsh WebGL Frogger qualification

DeepSeek Harness 0.1.5-rc.1 generated [`frogger.html`](frogger.html) through
the qualified DS41RT API on port 8000. The run used DeepSeek V4.1 Flash with
thinking enabled, high reasoning effort, a 65,536-token harness context and a
32,768-token per-step output allowance. The artifact is one 15,763-byte HTML
file with no external assets, libraries, network calls or build step.

**[Play the generated game](https://tpurtell.github.io/ds41rt/frogger.html).**
Use the arrow keys, WASD or the on-screen direction pad. The game renders its
lanes, vehicles, logs, turtles and frog directly through WebGL2 with a WebGL1
fallback and inline shaders; it never requests a Canvas 2D context.

## Measured execution

| Measurement | Result |
|---|---:|
| Successful wall time | 433.17 s |
| Model steps | 15 |
| Tool calls | 16 |
| First file write | 222.7 s |
| Generated output tokens across steps | 33,469 |
| Cache-read tokens across steps | 486,939 |
| Final model context | 43,347 tokens |
| Agent logic simulation | 20/20 passed |
| Independent static checks | 20/20 passed |
| Chrome WebGL render | passed |

The sixteen calls comprise one initial write, eleven shell checks and four
edits. The agent found and repaired two related moving-platform defects during
its own simulation: carry needed a swept overlap and position delta, and a
wrapped platform needed to reset its previous position so the teleport did not
look like a sweep across the board. Its final stubbed DOM/WebGL simulation
passed collision, drowning, diving-turtle, log-carry, goal, win, game-over,
restart and 3,000-frame finite-state checks.

The first qualification attempt is retained as a failure. With a 16,384-token
output allowance it spent 252.83 seconds designing the game before any write,
then ended at `max-tokens`; its only tool call had inspected the empty workspace.
The successful retry kept thinking enabled at high effort, doubled the output
allowance, and explicitly required an immediate compact write. It still spent
222.7 seconds before the first write, but then completed the full repair loop.

Independent validation parsed the extracted JavaScript with Node, checked the
single-file and offline contract, and verified all requested controls and game
systems. Google Chrome rendered the exact final artifact at 1280 x 900 through
software WebGL with exit status zero and no browser errors. The visible frame
contains the HUD, five goal slots, five river lanes, five road lanes, start
overlay and accessible pointer controls.

The run used the same clean-built corrected-FP4 coordinator and Spark images as
the release performance campaign. The machine-readable record is
[`release-v1-frogger.json`](release-v1-frogger.json); prompts, overlays, complete
harness sessions, logs, checks and the browser screenshot are preserved in
[`evidence/native-release-frogger.tar.gz`](evidence/native-release-frogger.tar.gz).

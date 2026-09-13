# V2 high-thinking tool evaluation

Three `tool-eval-bench` hard-mode campaigns exercised the clean v2 dSpark
images at C16. Every request used thinking enabled, high reasoning effort,
temperature zero, a 900-second timeout, twelve allowed tool turns, and the
server's normal output policy without an added token cap.

| Run | Basic | Hard | Total | Pass / partial / fail |
|---:|---:|---:|---:|---:|
| 1 | 122/138 | 33/38 | 155/176 | 71 / 13 / 4 |
| 2 | 118/138 | 35/38 | 153/176 | 69 / 15 / 4 |
| 3 | 122/138 | 36/38 | 158/176 | 73 / 12 / 3 |
| Mean | 120.67/138 | 34.67/38 | 155.33/176 | — |

All 264 scenarios completed, and the qualification driver reported success.
There were no request timeouts, worker failures, or serving-process failures.
The final standard service remained healthy after the campaign.

`TC-43` and `TC-68` failed all three runs: the model supplied an empty required
search query in the first case and chose a tool where none was needed in the
second. `TC-61` failed twice. `TC-21`, `TC-51`, and `TC-88` each failed once.
These are retained as model-quality outcomes; no run or scenario was retried.

The coordinator used an RTX PRO 6000 Blackwell Workstation Edition at the
enforced 400 W power limit and standard 14,001 MHz maximum memory speed. The
clean images identify source revision
`9477b6e39f4bbe431c4cd6d48c8b303045f9238f`, model revision
`dba1be0a40aa45a94ad051997016db3960a90277`, and SparkInfer revision
`bae6e5cf08fc7e51e8ea40f287dcfa95440036fc`.

The [v2 performance JSON](release-v2-performance.json) contains all three run
summaries and failures. The release evidence archive preserves each complete
result, raw trace, benchmark database, command, and log; its summary source has
SHA-256 `94f45282e9c84fab82d001825e2d70728429ef0a400b39252b3bd9ebe11083b8`.

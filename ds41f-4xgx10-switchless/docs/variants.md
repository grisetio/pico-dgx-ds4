# Weight variants

The engine can hold a second set of one tensor family beside the weights it binds, and serve each request
with either. The case it was made for: [pqhaz/DeepSeek-V4.1-Flash-1017-heretic][heretic] (MIT), an abliteration
of the official 1017 weights that changes only `layers.{2..39}.attn.wo_b.weight` (38 FP8 e4m3 tensors; their
block scales and every other tensor are byte-identical), published as a 1.6 GB patch over the official files.
The official weights stay the default; a client switches by naming another model.

[heretic]: https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic

## How it runs

- **Rank file.** `dsw_convert --ckpt DIR --rank R --only attn.wo_b.weight --no-coverage --out rankR.heretic.dsw`,
  with DIR holding only the patch's `heretic_patch.safetensors`: each rank's [5120][2048] column slice of the 38
  tensors, L1-tiled like the main file's, ~400 MB per rank. The engine also accepts them untiled (it tiles them at
  load, as `bind` does).
- **Engine.** `--variant FILE`, or the fleet file's `RANK<r>_VARIANT`, on every rank (serve only): the file's
  tensors are loaded into their own arena and checked against the bound ones' shapes and layers
  (`Model::bind_variant`). `SubmitHeader::variant` (the former reserved field; 0 for every older front-end) names
  the weights a request runs on; one the engine lacks is refused with an error.
- **Switch.** Requests run with one set at a time. A request for the other set waits until nothing is in flight,
  paused or waiting, and the requests after it wait their turn behind it, so neither model starves the other. Then
  rank 0 sends `kCmdVariant` and every rank exchanges the live slices with the held ones (three device copies per
  layer through a 10 MB buffer) and gathers the prefill path's copy of every rank's slice again
  (`Model::set_variant`). The tensors' addresses do not change, so the decode graphs, the L2 prefetch and the
  prefill stay valid, and an exchange back restores the bytes exactly. Rank 0 logs `serve: weight variant N bound
  in X ms`.
- **Sessions.** A session's state is its weights': the server tags each session with the variant it was computed
  with and never continues or forks a session of the other (it starts cold and frees the old one). The kept-session
  index (`DSGB10S2`) records the tag; an `S1` index reads as variant 0.
- **dsfront.** `--variant-model NAME`: requests whose `model` is NAME run on the variant, every other name on the
  bound weights. `/v1/models` lists both names, responses carry the one served, and the session table matches a
  prompt only against the sessions of its own variant (`DSFT0002`; a `DSFT0001` table reads as variant 0).

## Deploying

1. Download the patch (`patch/heretic_patch.safetensors` at a pinned revision) and check its tensor list: the 38
   `attn.wo_b.weight` names, F8_E4M3, nothing else.
2. Convert per rank as above. To check the conversion, convert the official checkpoint's tensors the same way
   (`--only attn.wo_b.weight`) and compare them with the main rank file's.
3. Copy each rank's file to its box, add `RANK<r>_VARIANT=~/dsgb10/weights/rank{rank}.heretic.dsw` to the fleet
   file and `--variant-model NAME` to dsfront's arguments, and restart.
4. Check: a greedy request on the default model gives the same output as before the change; the same request on
   NAME differs; the default again gives the first output.

## Limits

- One variant, of `attn.wo_b.weight` only: another tensor family needs its readers' derived copies found and
  refreshed the way `set_variant` gathers the prefill's.
- A switch drains the engine first, so mixed traffic of both models runs in alternating batches.
- The drafter (DSpark/MTP) is the bound model's; its acceptance may be lower on the variant (speed only: drafts
  never change the tokens).

Another model family on the same endpoint, GLM-5.3 on gr8, with engine switching between the families and
the same waiting rule: [glm.md](glm.md).

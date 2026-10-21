# SmolVLM 256M model dependencies

No model weights are committed. This contribution does not create quantized models.

- Revised No-MediaPipe: load the pretrained SmolVLM-256M-Instruct base, then the matching selected adapter and processor.
- Dual-crop baseline and MediaPipe: the notebook first merges an earlier adapter into the base and then applies the selected vision-and-text adapter. Loading only the latter on an untouched base does not reproduce that initialization. Record the exact earlier adapter, not just a directory name.
- Keep the selected `best_by_macro_f1` adapter separate from `last_adapter` and optimizer-resume checkpoints.
- Preserve matching prompt text, crop preprocessing, processor and MediaPipe cache/task version. The downloaded helper snapshots are not proof of all historical versions.

The existing export CLI was written for the other model workflow and is not certified here for the chained 256M initialization. Use the verified notebook loading chain before implementing a portable export. GPU execution and export equivalence have not been tested in this repository update.

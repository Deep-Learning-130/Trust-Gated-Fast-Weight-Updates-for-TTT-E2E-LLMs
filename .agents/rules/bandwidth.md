---
description: Constraints for downloading large files, models, or datasets
---
# Resource and Bandwidth Limits

Do not download large models, datasets, checkpoints, or other large files without explicitly telling the user first and getting their approval.

## Specific Constraints:
1. **Approval required:** Before starting any download likely to exceed 100 MB, STOP and tell the user:
   - exactly what will be downloaded
   - approximate download size
   - why it is needed
   - whether there is a smaller/local alternative
   - whether the download is required for the current task or merely convenient
2. **No convenience downloads:** Do not automatically download alternative model sizes, reference models, checkpoints, datasets, or duplicate copies just to perform comparisons.
3. **Implicit downloads:** Do not download a model merely because a script calls `from_pretrained()` or because it would make validation easier. If a command may trigger an indirect download (for example through Hugging Face Transformers or Datasets), treat that as a download and ask first. If you are unsure whether something will download a large file, assume it might and ask the user before executing it.
4. **Cache reuse:** Check the local Hugging Face cache first. Reuse already-downloaded files whenever possible. If a required file is not cached, stop before downloading it and ask the user.
5. **No concurrency:** Never start multiple large downloads concurrently.
6. **No assumptions:** Do not interpret "continue working" or "do everything you can" as permission to download large files. Small normal dependency/package downloads are fine, but anything potentially large must be disclosed first.
7. **P3 Specific:** For the current P3 work in particular, do not download any additional model or dataset until you tell the user exactly what is needed and they approve it.

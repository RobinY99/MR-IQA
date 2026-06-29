# Data Directory

Place public or locally prepared manifests here. Do not commit private images, large datasets, or generated outputs.

The committed manifests keep image paths relative to the runtime `IMAGE_ROOT`; they should not contain machine-specific absolute paths.

Recommended layout:

```text
manifest_checksums.json
train_manifest/train.jsonl
val_manifests/koniq_val_200_seed42.json
test_manifests/agiqa3k.json
test_manifests/csiq.json
test_manifests/kadid_full.json
test_manifests/koniq.json
test_manifests/livew.json
test_manifests/pipal.json
test_manifests/spaq_full.json
test_manifests/tid2013.json
```

# hdr10plus_tool

CLI utility to work with **HDR10+ dynamic metadata in AV1 video files**.

> ⚠️ **IMPORTANT**
>
> This tool is **ONLY for AV1 HDR10+**.
> **HEVC is NOT supported.**
> **Only IVF containers are supported.**

---

## Supported Format

* **Video codec:** AV1
* **Container:** IVF
* **Metadata:** HDR10+ (ITU-T T.35, SMPTE ST 2094-40, Profile B)

---

## Usage

```bash
python hdr10plus_tool.py <command> [options]
```

Available commands:

* `extract`
* `inject`
* `remove`
* `plot`

---

## Commands

### extract

Extract HDR10+ metadata from an **AV1 IVF** file into a classic HDR10+ JSON file.

```bash
python hdr10plus_tool.py extract -i input.ivf -o metadata.json
```

---

### inject

Inject HDR10+ metadata from a classic JSON file into an **AV1 IVF** file.

* Existing HDR10+ metadata is removed before injection.
* If video and JSON lengths differ:

  * Shorter video → metadata is truncated
  * Longer video → metadata is duplicated

```bash
python hdr10plus_tool.py inject -i input.ivf -j metadata.json -o output.ivf
```

---

### remove

Remove HDR10+ metadata from an **AV1 IVF** file.

```bash
python hdr10plus_tool.py remove -i input.ivf -o output.ivf
```

---

### plot

Generate a PNG plot from HDR10+ metadata stored in a JSON file.

```bash
python hdr10plus_tool.py plot -i metadata.json -o plot.png
```

---

## Notes

* This tool works **exclusively** with **AV1 HDR10+**.
* Files without HDR10+ metadata will cause extraction to fail.
* JSON format is compatible with classic HDR10+ workflows.

---

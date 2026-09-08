# Performance notes

Everything below was found by reading the checked-in pipeline source and the
pinned upstream model packages under `uber_model/huggingface/`. Each item cites
the file and line it came from. Items are ordered by expected impact.

The recurring theme is that the individual models are already well optimised on
the GPU. The cost the fused model removes is almost entirely at the *seams*
between them: the production pipeline hands page images between Ray actors as
base64 strings in pandas columns, so every stage re-decodes the same image and
copies it across PCIe again.

## 1. OCR forces a full device round-trip on its own input

This is the single largest win.

`nemotron_ocr` v2 is internally excellent: preprocess, detector, centerness
prefilter, NMS, rectify plus grid sample, recogniser, and the relational model
all run on device. But its entry point discards device residency:

```282:283:uber_model/huggingface/nvidia--nemotron-ocr-v2/nemotron-ocr/src/nemotron_ocr/inference/pipeline_v2.py
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
```

`_preprocess_batch` then copies it straight back:

```253:254:uber_model/huggingface/nvidia--nemotron-ocr-v2/nemotron-ocr/src/nemotron_ocr/inference/pipeline_v2.py
            tensor_gpu = tensor.to("cuda", non_blocking=True)
            tensor_gpu = tensor_gpu.to(torch.float16).div_(255.0)
```

So handing `nemotron_ocr` a CUDA tensor costs a device-to-host copy plus a
host-to-device copy, for no benefit.

The NRL wrapper makes it considerably worse. `NemotronOCRV2.invoke` converts a
torch tensor by encoding it as a PNG and base64ing it:

```134:158:nemo_retriever/src/nemo_retriever/models/local/nemotron_ocr_v2.py
        x = img.detach()
        if x.device.type != "cpu":
            x = x.cpu()
        ...
            arr = x.squeeze(0).numpy()   # or permute(1,2,0).numpy()
            pil = Image.fromarray(arr, mode="RGB")
        ...
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
```

which upstream then base64-decodes and torchvision-decodes back onto the device.
The full path for one crop is: device tensor, host tensor, numpy array, PIL
image, PNG encode, base64 encode, base64 decode, PNG decode, host tensor,
device tensor. That is two PCIe crossings, a lossless-compression encode, and a
decode, all to move data that was already in the right memory.

**Fused approach.** `GpuOcrStage` skips `_load_image_to_tensor_uint8` and
`_preprocess_batch`, builds the detector batch on device with
`gpu_ops.resize_and_pad_for_ocr`, and calls the upstream phase methods
(`_run_detector_batched`, `_prefilter_detections`, `_run_nms`,
`_run_rectify_and_sample`, `_run_recognizer_chunked`) directly. The model weights
and inference graph are untouched; only the host-side entry point is bypassed.

**Expected effect.** Eliminates the PNG encode and decode entirely, and removes
two PCIe crossings per OCR crop. A page with 20 text and table regions pays
this 20 times per page in the production path.

## 2. Page images are re-decoded once per stage

Page images travel between Ray actors as base64 in the `page_image.image_b64`
column, so each stage decodes independently:

* Page elements decodes to a BGR numpy array in
  `_decode_b64_image_to_np_array` (`common/modality/page_elements/shared.py:115`).
* OCR decodes the same page again to crop regions in `_crop_all_from_page`
  (`common/modality/ocr/shared.py`).
* Table structure crops from another decode of the same page.
* Embed decodes it a fourth time through `_b64_to_pil`
  (`models/local/llama_nemotron_embed_vl_1b_v2_embedder.py:31`).

**Fused approach.** `DeviceImageBatch` decodes once. For JPEG payloads the
decode itself runs on the GPU via nvJPEG (`torchvision.io.decode_jpeg`), so the
only thing crossing PCIe is the compressed byte buffer, which is roughly 10 to
20 times smaller than the decoded pixels. Every later stage reads the same
resident tensor.

**Expected effect.** Four decodes become one, and one host-to-device pixel
transfer per page becomes one compressed-bytes transfer.

## 3. Region cropping goes through PIL and base64

`_crop_all_from_page` decodes the page, crops with PIL, and for the remote path
re-encodes each crop as base64 PNG:

```214:220:nemo_retriever/src/nemo_retriever/common/modality/ocr/shared.py
def _np_rgb_to_b64_png(crop_array: np.ndarray) -> str:
    ...
    return base64.b64encode(buf.getvalue()).decode("ascii")
```

**Fused approach.** `gpu_ops.crop_regions` slices the resident page tensor.
A slice shares storage with the page, so cropping allocates nothing and copies
nothing — the CUDA residency test asserts `crops[0].data_ptr() == image.data_ptr()`.

**Expected effect.** Region cropping drops from a decode, N PIL crops, and N PNG
encodes to N pointer offsets.

## 4. Weighted box fusion runs in numpy on the host

Page-elements NMS runs on device inside the upstream YOLOX `postprocess`, but
the results are then pulled to the host so the numpy WBF pass can run:

```132:134:uber_model/huggingface/nvidia--nemotron-page-elements-v3/nemotron_page_elements_v3/utils.py
    boxes = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
```

```201:204:nemo_retriever/src/nemo_retriever/common/modality/page_elements/shared.py
        # Move to CPU for safe conversion.
        bi = bi.detach().cpu()
        li = li.detach().cpu()
        si = si.detach().cpu()
```

followed by per-detection scalar reads:

```218:224:nemo_retriever/src/nemo_retriever/common/modality/page_elements/shared.py
                x1, y1, x2, y2 = [float(x) for x in bi[j].tolist()]
            ...
                label_i = int(li[j].item())
            ...
                score_f = float(si[j].item())
```

The `.cpu()` calls are the real cost, not the bytes: each one synchronises the
stream, so the GPU idles until the host catches up, right at the point where the
next stage's work could have been queued.

**Fused approach.** `gpu_ops.weighted_box_fusion` and
`gpu_ops.apply_per_class_thresholds` run in torch on device. Detections stay
resident all the way through cropping, and the only host read is a single
batched `.tolist()` per page when assembling results.

**Expected effect.** Removes one synchronisation per page from the critical path,
and replaces the numpy pairwise loop with a single IoU matrix.

## 5. VL embedding preprocesses on the host with PIL

The VL processor resizes with PIL, crops up to `max_num` tiles with PIL, and runs
`ToTensor` plus `Normalize` per tile:

```149:165:uber_model/huggingface/nvidia--llama-nemotron-rerank-vl-1b-v2/processing_llama_nemotron_vl.py
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        ...
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    ...
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
```

For a six-tile page that is one PIL bicubic resize of a full page, seven crops,
and seven separate normalisation passes, on the host, for an image the OCR stage
already had on the GPU.

**Fused approach.** `gpu_ops.tile_for_vl_tower` does the resize with
`F.interpolate(mode="bicubic")`, produces the tiles with a reshape and permute
instead of crops, and folds the 0-255 scaling, mean subtraction, and standard
deviation division into one elementwise expression.

**Expected effect.** The tiling becomes one interpolate plus one permute copy
instead of seven PIL operations and seven normalisation passes, and it happens
on the device that already holds the pixels.

## 6. The two detectors are the same architecture and can overlap

Page-elements and table-structure are byte-for-byte the same YOLOX geometry.
Compare the two `Exp.get_model` bodies:

```71:79:uber_model/huggingface/nvidia--nemotron-page-elements-v3/nemotron_page_elements_v3/page_element_v3.py
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act
            )
            self.model = YOLOX(backbone, head)
```

```64:72:uber_model/huggingface/nvidia--nemotron-table-structure-v1/nemotron_table_structure_v1/table_structure_v1.py
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act
            )
            self.model = YOLOX(backbone, head)
```

Both use depth 1.0, width 1.0, and a 1024x1024 input. They differ only in class
count (6 versus 5) and NMS parameters (iou 0.5 class-agnostic versus 0.25
class-aware).

**Fused approach.** One `FusedYoloxDetector` implementation and one
`gpu_ops.letterbox_batch` preprocessing kernel serve both stages. Because the
graphs have identical shapes and no data dependency once the page tensor exists,
table structure is issued on a second CUDA stream and interleaves with the
page-elements work.

**Expected effect.** Two identically shaped graphs interleave rather than
serialise, and cuDNN autotuning results are shared because the kernel shapes
match.

## 7. Table structure was capped at batch size one

The local wrapper discards everything past the first batch element:

```nemo_retriever/src/nemo_retriever/models/local/nemotron_table_structure_v1.py
    def invoke(self, input_tensor, orig_shape):
        return self._model(...)[0]
```

So even though the upstream forward pass supports batching, a page with six
tables ran six separate forward passes, and the pipeline loops per crop.

**Fused approach.** `TableStructureStage.run` takes every table crop in the
invocation, letterboxes them into one batch, and runs a single forward pass. The
crops are flattened across all pages in the batch, not just within one page.

**Expected effect.** A 16-page batch with three tables per page goes from 48
forward passes to a handful of batched ones.

## 8. Every stage boundary is a synchronisation

Ray actors exchange pandas DataFrames through the object store, which means each
stage boundary is a serialise, transfer, deserialise, and an implicit wait for
the previous stage's host-side data. The pipeline cannot queue stage N+1's work
while stage N is still running.

**Fused approach.** All stages are issued into the same CUDA context and
`forward` synchronises exactly once, after everything is queued. Stage timings
use CUDA events rather than `time.perf_counter`, so measuring costs nothing and
does not itself introduce a synchronisation.

**Expected effect.** Removes three inter-stage synchronisations per batch and
lets the driver keep the GPU fed across stage boundaries.

## 9. Smaller wins already applied

* **Letterbox writes into a pre-filled buffer.** `gpu_ops.letterbox_batch`
  allocates the padded batch once at the pad value and writes each resized image
  into its slice, instead of `F.pad` per image followed by `torch.stack`. Saves
  one allocation and one copy per page.
* **Batched host reads.** Where a host read is genuinely needed, it is one
  `.tolist()` for the whole tensor rather than per-element `.item()` calls. Each
  `.item()` is its own synchronisation.
* **In-place normalisation in the target dtype.** `resize_and_pad_for_ocr`
  divides and casts in one pass rather than materialising an intermediate
  float32 tensor.
* **TF32 and cuDNN autotuning enabled once** at load, rather than per stage.
* **Device-side top-k for reranking.** `NemoRetrieverFusedModel.rerank` sorts and
  truncates on device, so only the surviving candidates are ever copied to the
  host.

## Remaining opportunities, not yet implemented

These are recorded rather than done, because each needs measurement on real
hardware to justify the complexity.

1. **CUDA graph capture for the detectors.** The `ExecutionConfig.graph_capture`
   and `static_batch_size` fields exist but the capture is not wired up. YOLOX at
   a fixed 1024x1024 batch is an ideal candidate: the shape never varies, so
   capture would remove per-launch overhead across roughly 200 kernels. This
   needs padding every batch to a static size, which wastes compute on partial
   batches, so it is a throughput-versus-latency trade that should be measured.
2. **TensorRT for both YOLOX detectors.** The OCR wrapper already has an opt-in
   `torch_tensorrt` path for its detector behind `RETRIEVER_ENABLE_TORCH_TRT`.
   The same treatment for the two YOLOX models is straightforward since their
   input shape is static.
3. **Fusing the two detector heads.** Because the backbones are architecturally
   identical, a single batched forward over concatenated inputs is not possible
   with separate weights, but the two backbones could share a CUDA graph and the
   heads could be run as a grouped convolution. This is speculative.
4. **GPU table assembly.** `table_assembly.py` runs on the host, which is correct
   for string work, but the row and column band assignment is numeric and could
   move to device, leaving only the final string join on the host.
5. **nvJPEG batched decode.** `decode_base64_image` decodes one page at a time.
   `torchvision.io.decode_jpeg` accepts a list and batches the nvJPEG calls,
   which would help when a batch has many pages.
6. **Keeping embeddings on device through the vector database write.**
   `EmbedResult.vectors` stays resident, but the current consumers all call
   `to_host()`. A writer that accepts device buffers would remove the last
   transfer in the ingest path.

## How to measure

```bash
export NEMO_RETRIEVER_FUSED_MIRROR=/path/to/uber_model/huggingface

# Fused pipeline, with the transfer accounting the benchmark installs.
python -m nemo_retriever_fused.benchmark --synthetic 16 --iterations 5

# Isolate the effect of individual optimisations.
python -m nemo_retriever_fused.benchmark --synthetic 16 --no-overlap
python -m nemo_retriever_fused.benchmark --synthetic 16 --no-device-decode
```

The benchmark reports per-stage device milliseconds from CUDA events plus
host-to-device and device-to-host byte counts. The byte counts are the more
portable comparison against the production pipeline, since they do not vary with
GPU model.

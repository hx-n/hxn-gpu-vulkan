# hxn-gpu-vulkan

[![Ash](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/ash.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/ash.yml)
[![HashLink](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hashlink.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hashlink.yml)
[![hxcpp](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hxcpp.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hxcpp.yml)
[![CPPIA](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/cppia.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/cppia.yml)

Vulkan for Haxe on Ash, HashLink, hxcpp and CPPIA, on Linux, Windows, Android
and macOS (through MoltenVK). It keeps Vulkan's own conventions: an instance
finds the GPUs, a device makes buffers, images, descriptor sets and
pipelines, and command buffers record work that a program submits.
Synchronization is explicit: the program records the barriers and image
layout transitions its commands need, as in Vulkan.

Shaders come from [hxnsl](https://github.com/hx-n/hxnsl), which compiles HXSL
to SPIR-V during the Haxe build. hxnsl's bind groups are the descriptor sets,
and its `LAYOUT` gives their layouts, so nothing reflects on a shader at run
time.

```haxe
import gpu.vulkan.*;

var instance = Instance.create("my app");
var device = instance.createDevice(0);
var module = device.createShaderModule(Tripled.spirv());          // an hxnsl shader
var sets = gpu.vulkan.Shaders.setLayouts(device, Tripled.LAYOUT);
var layout = device.createPipelineLayout(sets);
var pipeline = device.createComputePipeline(module, "compute_main", layout);
var data = device.createBufferWith(input, BufferUsage.Storage);
var set = device.createDescriptorSet(sets, Tripled.BUFFER_data_GROUP);
set.writeBuffer(Tripled.BUFFER_data, data, 0, input.length);

var commands = device.createCommandBuffer();
commands.bindPipeline(BindPoint.Compute, pipeline);
commands.bindDescriptorSet(BindPoint.Compute, layout, Tripled.BUFFER_data_GROUP, set);
commands.dispatch(4, 1, 1);
commands.bufferBarrier(data, Stage.ComputeShader, Access.ShaderWrite, Stage.Host, Access.HostRead);
device.submit(commands).then(_ -> trace(data.read(0, input.length)));
```

[`examples/check/Main.hx`](https://github.com/hx-n/hxn-gpu-vulkan/blob/main/examples/check/Main.hx)
is the whole program, with a render pass into an image that is copied back
as well.

## Installing

```
haxelib install hxn-gpu-vulkan
haxelib install hxnsl
haxe -lib hxn-gpu-vulkan -lib hxnsl -main Main -hl main.hl
haxe -lib hxn-gpu-vulkan -lib hxnsl -main Main -cpp out
```

The haxelib carries the native library prebuilt for Linux on x86-64 and
ARM64, Windows on x86-64 and macOS on Apple silicon, for HashLink and hxcpp,
and as static libraries for hxcpp on Intel Macs and for Android. A program
needs no Rust. It needs a Vulkan driver and loader: on Linux, Mesa's drivers
(`libvulkan1 mesa-vulkan-drivers`); on macOS, MoltenVK from the Vulkan SDK.

## Classes

All in the package `gpu.vulkan`:

| Class | |
|---|---|
| `Instance` | `create(name)`, the GPUs (`adapterNames`), and `createDevice(index)`. |
| `Device` | Buffers, images, samplers, shader modules, set layouts, pipeline layouts, descriptor sets, pipelines and command buffers; `submit` and `submitAndWait`. |
| `Buffer` | Memory the CPU maps: `write`, `read`. |
| `Image` | A 2D image in device memory, described by an `ImageDescriptor`, with one view of all of it. |
| `SetLayouts`, `PipelineLayout` | The descriptor set layouts of a pipeline, and the pipeline layout of them. |
| `DescriptorSet` | `writeBuffer`, `writeImage`, `writeSampler` by binding. |
| `Pipeline` | A compute or graphics pipeline. A graphics pipeline renders into one color attachment with dynamic rendering, and takes its vertex buffers' layout as `VertexBinding`s and `VertexAttribute`s. |
| `CommandBuffer` | Binding, dispatches, rendering, draws, copies, and `bufferBarrier`, `imageBarrier` and `memoryBarrier`. |

Every object holds its Vulkan object until `close()`, and the device lives
until everything made from it is closed. Vulkan's errors are thrown as Haxe
exceptions.

## Synchronization

Barriers take stage and access bits (`Stage`, `Access`, or-ed together) for
the work before and the work after, and an image barrier also moves the image
between `Layout`s. The check shows the usual ones: a compute shader's writes
made visible to the CPU, an image moved to the color attachment layout before
rendering, then to the transfer source layout for a copy.

`Device.submit` returns an `hxn.Future` that settles when the GPU has
finished, from a thread that waits on the submission's fence; the command
buffer is kept until then. `submitAndWait` is the blocking form.

## Shaders from hxnsl

A shader for this driver implements `gpu.vulkan.Shader`, which extends
`hxnsl.Shader` and gets hxnsl's `spirv()` and `LAYOUT`. SPIR-V's entry points
keep hxnsl's names: `vertex_main`, `fragment_main` and `compute_main`.
`gpu.vulkan.Shaders.setLayouts(device, Shader.LAYOUT)` makes the descriptor
set layouts of its bind groups.

russl, which writes the SPIR-V, flips Y in the vertex stage, so images come
out the way WebGPU draws them with an ordinary viewport.

## Building from source

Requires the [hxn](https://github.com/hx-n/hxn) CLI, Rust, Haxe 4.3, hxnsl
and a Vulkan driver. `hxn check` builds the binding and runs `examples/check`
on each runtime named, and fails unless every run passes and prints the same
output. CI runs it on Linux with lavapipe and the Khronos validation layer:

```
hxn check hl --vm ash
hxn check cpp cppia
```

## Agent skill

[`skills/hxn-gpu-vulkan/SKILL.md`](https://github.com/hx-n/hxn-gpu-vulkan/blob/main/skills/hxn-gpu-vulkan/SKILL.md)
describes the library for coding agents. The haxelib includes it.

## License

Apache-2.0.

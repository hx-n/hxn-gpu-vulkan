# hxn-gpu-vulkan

[![Ash](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/ash.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/ash.yml)
[![HashLink](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hashlink.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hashlink.yml)
[![hxcpp](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hxcpp.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/hxcpp.yml)
[![CPPIA](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/cppia.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/cppia.yml)
[![Present](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/present.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-vulkan/actions/workflows/present.yml)

Vulkan bindings for Haxe on Ash, HashLink, hxcpp and CPPIA, on Linux,
Windows, Android and macOS (through MoltenVK). The API follows Vulkan:
`Instance`, `Device`, buffers, images, descriptor sets, pipelines, command
buffers, and swapchains that present to a window. Synchronization is
explicit: you record the barriers and image layout transitions your commands
need.

Shaders come from [hxnsl](https://github.com/hx-n/hxnsl), which compiles HXSL
to SPIR-V at compile time. hxnsl's bind groups are the descriptor sets, and
`LAYOUT` gives their layouts, so no shader reflection happens at run time.

```haxe
import hxn.gpu.vulkan.*;

var instance = Instance.create("my app");
var device = instance.createDevice(0);
var module = device.createShaderModule(Tripled.spirv());          // an hxnsl shader
var sets = hxn.gpu.vulkan.Shaders.setLayouts(device, Tripled.LAYOUT);
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

All in the package `hxn.gpu.vulkan`:

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
| `Surface` | A window's surface, from `Instance.createSurface(platform, raw0, raw1, raw2, raw3)`. |
| `Swapchain` | A window's images, from `Device.createSwapchain`: `acquire`, `present` and `resize`. |

Every object holds its Vulkan object until `close()`, and the device lives
until everything made from it is closed. Vulkan's errors are thrown as Haxe
exceptions.

## Windows

`Instance.createSurface` takes [hxn-window](https://github.com/hx-n/hxn-window)'s
`platform()` and `raw(0)` to `raw(3)`, on AppKit and UIKit (through MoltenVK),
Win32, Xlib, Wayland and Android. `Device.createSwapchain(surface, width, height,
format, vsync)` makes the images to present. Each frame, `acquire` waits for the
next image, a render pass draws into it, a barrier moves it to `Layout.Present`,
and `present` shows it once the work submitted before has finished. `present`
returns false when the window has changed size; `resize` then makes the
swapchain again. The window's surface sets the images' size, which on a
high-density display can be larger than the size asked for: read it from
`width` and `height`.

```haxe
var surface = instance.createSurface(window.platform(), window.raw(0), window.raw(1), window.raw(2), window.raw(3));
var swapchain = device.createSwapchain(surface, 640, 480, Format.Bgra8Unorm, true);
var image = swapchain.acquire();
commands.imageBarrier(image, Stage.TopOfPipe, 0, Layout.Undefined, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite, Layout.ColorAttachment);
commands.beginRendering(image, LoadOp.Clear, 0, 0, 0, 1);
// ... draws ...
commands.endRendering();
commands.imageBarrier(image, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite, Layout.ColorAttachment, Stage.BottomOfPipe, 0, Layout.Present);
device.submit(commands);
if (!swapchain.present())
	swapchain.resize(width, height);
```

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

A shader for this driver implements `hxn.gpu.vulkan.Shader`, which extends
`hxnsl.Shader` and gets hxnsl's `spirv()` and `LAYOUT`. SPIR-V's entry points
keep hxnsl's names: `vertex_main`, `fragment_main` and `compute_main`.
`hxn.gpu.vulkan.Shaders.setLayouts(device, Shader.LAYOUT)` makes the descriptor
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

## Third-party notices

Dependency licenses and redistribution notices are in
[third-party/NOTICES.md](https://github.com/hx-n/hxn-gpu-vulkan/blob/main/third-party/NOTICES.md). Retain the applicable
notices when redistributing binaries.

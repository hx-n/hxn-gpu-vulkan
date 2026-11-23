import hxn.gpu.vulkan.*;
import haxe.io.Bytes;
import Shaders;

/**
	What CI runs: a compute shader over a buffer, and render passes into an
	image, without blending and with alpha blending, each read back and checked, with the barriers Vulkan needs
	between the GPU's writes and the CPU's reads. It needs a Vulkan driver,
	and prints the same lines on every runtime.
**/
class Main {
	static function check(ok:Bool, message:String) {
		if (!ok)
			throw message;
	}

	static function closeAll(objects:Array<{function close():Void;}>) {
		for (o in objects)
			o.close();
	}

	static function main() {
		var instance = try Instance.create("hxn-gpu-vulkan check") catch (e:haxe.Exception) {
			Sys.println('no Vulkan: ${e.message}');
			return;
		}
		// A real GPU when there is one, else the first device, such as a software one.
		var names = instance.adapterNames();
		var chosen = 0;
		for (i in 0...names.length)
			if (names[i].indexOf("llvmpipe") < 0) {
				chosen = i;
				break;
			}
		var device = instance.createDevice(chosen);
		// The compute pass completes in a callback from the Haxe event loop, after main
		// returns, and the render pass runs from there.
		compute(device, () -> {
			render(device);
			device.close();
			instance.close();
		});
		Sys.println("compute: submitted");
	}

	static function compute(device:Device, next:() -> Void) {
		var module = device.createShaderModule(Tripled.spirv());
		var sets = hxn.gpu.vulkan.Shaders.setLayouts(device, Tripled.LAYOUT);
		var layout = device.createPipelineLayout(sets);
		var pipeline = device.createComputePipeline(module, "compute_main", layout);
		var count = 64;
		var input = Bytes.alloc(count * 4);
		for (i in 0...count)
			input.setInt32(i * 4, i);
		var data = device.createBufferWith(input, BufferUsage.Storage);
		var set = device.createDescriptorSet(sets, Tripled.BUFFER_data_GROUP);
		set.writeBuffer(Tripled.BUFFER_data, data, 0, count * 4);

		var commands = device.createCommandBuffer();
		commands.bindPipeline(BindPoint.Compute, pipeline);
		commands.bindDescriptorSet(BindPoint.Compute, layout, Tripled.BUFFER_data_GROUP, set);
		commands.dispatch(Std.int(count / Tripled.ENTRY_POINTS[0].workgroup[0]), 1, 1);
		// The kernel's writes, made visible to the CPU's read.
		commands.bufferBarrier(data, Stage.ComputeShader, Access.ShaderWrite, Stage.Host, Access.HostRead);
		device.submit(commands).then(_ -> {
			var output = data.read(0, count * 4);
			for (i in 0...count)
				check(output.getInt32(i * 4) == i * 3 + 1, 'value $i is ${output.getInt32(i * 4)}, not ${i * 3 + 1}');
			Sys.println('compute: $count values tripled plus one, the last ${output.getInt32((count - 1) * 4)}');
			closeAll([commands, set, data, pipeline, layout, sets, module]);
			next();
		}, e -> {
			Sys.println('compute failed: ${e.message}');
			hxn.Loop.exit(1);
		});
	}

	// Blending is off: the quad's color replaces the target's.
	static final REPLACE:Blend = {
		enabled: false,
		colorOperation: 0,
		colorSource: 1,
		colorDestination: 0,
		alphaOperation: 0,
		alphaSource: 1,
		alphaDestination: 0
	};

	// Straight alpha: color = source * source alpha + destination * (1 - source alpha),
	// and alpha = source + destination * (1 - source alpha). The shared factor codes 4
	// (SrcAlpha), 5 (OneMinusSrcAlpha) and 1 (One), and the operation 0 (Add).
	static final ALPHA:Blend = {
		enabled: true,
		colorOperation: 0,
		colorSource: 4,
		colorDestination: 5,
		alphaOperation: 0,
		alphaSource: 1,
		alphaDestination: 5
	};

	static function render(device:Device) {
		var tint = [0.25, 0.5, 0.75, 1.0];
		var pixels = quad(device, tint, [0.0, 0.0, 0.0, 0.0], REPLACE);
		var expected = [for (c in tint) Math.round(c * 255)];
		for (p in 0...16)
			for (c in 0...4)
				check(pixels.get(p * 4 + c) == expected[c], 'pixel $p channel $c is ${pixels.get(p * 4 + c)}, not ${expected[c]}');
		Sys.println('render: 4x4 pixels of ${[for (c in 0...4) pixels.get(c)].join(", ")}');
		// A half-transparent red quad over a blue target: half red, half blue, opaque.
		var blended = quad(device, [1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0], ALPHA);
		var mixed = [128, 0, 128, 255];
		for (p in 0...16)
			for (c in 0...4)
				check(Math.abs(blended.get(p * 4 + c) - mixed[c]) <= 1, 'blended pixel $p channel $c is ${blended.get(p * 4 + c)}, not about ${mixed[c]}');
		Sys.println('blend: 4x4 pixels of about ${mixed.join(", ")}');
	}

	// Draws a quad of `tint` into a 4x4 target cleared to `clear`, with `blend`, and
	// returns the target's pixels.
	static function quad(device:Device, tint:Array<Float>, clear:Array<Float>, blend:Blend):Bytes {
		var module = device.createShaderModule(Tinted.spirv());
		var sets = hxn.gpu.vulkan.Shaders.setLayouts(device, Tinted.LAYOUT);
		var layout = device.createPipelineLayout(sets);
		var pipeline = device.createGraphicsPipeline(module, "vertex_main", module, "fragment_main", layout, Format.Rgba8Unorm, Topology.TriangleStrip,
			[{binding: 0, stride: 8, perInstance: false}], [{location: Tinted.INPUT_position, binding: 0, format: Format.Rg32Float, offset: 0}], blend);
		var size = 4;
		var target = device.createImage({
			width: size,
			height: size,
			format: Format.Rgba8Unorm,
			usage: ImageUsage.ColorAttachment | ImageUsage.TransferSrc
		});
		var readback = device.createBuffer(size * size * 4, BufferUsage.TransferDst);

		var quad = Bytes.alloc(4 * 8);
		var corners = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
		for (i in 0...corners.length)
			quad.setFloat(i * 4, corners[i]);
		var positions = device.createBufferWith(quad, BufferUsage.Vertex);
		var params = Bytes.alloc(Tinted.PARAMS_SIZE);
		for (i in 0...4)
			params.setFloat(Tinted.PARAMS_tint + i * 4, tint[i]);
		var uniforms = device.createBufferWith(params, BufferUsage.Uniform);
		var set = device.createDescriptorSet(sets, Tinted.PARAMS_GROUP);
		set.writeBuffer(Tinted.PARAMS_BINDING, uniforms, 0, Tinted.PARAMS_SIZE);

		var commands = device.createCommandBuffer();
		commands.imageBarrier(target, Stage.TopOfPipe, 0, Layout.Undefined, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite,
			Layout.ColorAttachment);
		commands.beginRendering(target, LoadOp.Clear, clear[0], clear[1], clear[2], clear[3]);
		commands.bindPipeline(BindPoint.Graphics, pipeline);
		commands.bindDescriptorSet(BindPoint.Graphics, layout, Tinted.PARAMS_GROUP, set);
		commands.bindVertexBuffer(0, positions, 0);
		commands.draw(4, 1, 0);
		commands.endRendering();
		// The rendered image, moved for the copy, then the copy made visible to the CPU.
		commands.imageBarrier(target, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite, Layout.ColorAttachment, Stage.Transfer,
			Access.TransferRead, Layout.TransferSrc);
		commands.copyImageToBuffer(target, readback);
		commands.bufferBarrier(readback, Stage.Transfer, Access.TransferWrite, Stage.Host, Access.HostRead);
		device.submitAndWait(commands);

		var pixels = readback.read(0, size * size * 4);
		closeAll([commands, set, uniforms, positions, readback, target, pipeline, layout, sets, module]);
		return pixels;
	}
}

import hxn.gpu.vulkan.*;
import hxn.window.Window;
import hxn.window.WindowAttributes;

/**
	Presents to a window: a swapchain on an hxn-window window, three frames
	each cleared to one color and presented, the last copied out and checked
	before it is shown.
**/
class Main {
	static function main() {
		var attributes = new WindowAttributes();
		attributes.title("hxn-gpu-vulkan present");
		attributes.width(64);
		attributes.height(64);
		var window = Window.open(attributes);
		attributes.close();
		var instance = Instance.create("hxn-gpu-vulkan present");
		var device = instance.createDevice(0);
		var surface = instance.createSurface(window.platform(), window.raw(0), window.raw(1), window.raw(2), window.raw(3));
		var swapchain = device.createSwapchain(surface, 64, 64, Format.Bgra8Unorm, false);
		var width = swapchain.width();
		var height = swapchain.height();
		var readback = device.createBuffer(width * height * 4, BufferUsage.TransferDst);
		var colors = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
		for (frame in 0...colors.length) {
			var c = colors[frame];
			var image = swapchain.acquire();
			var commands = device.createCommandBuffer();
			commands.imageBarrier(image, Stage.TopOfPipe, 0, Layout.Undefined, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite, Layout.ColorAttachment);
			commands.beginRendering(image, LoadOp.Clear, c[0], c[1], c[2], 1);
			commands.endRendering();
			var last = frame == colors.length - 1;
			if (last) {
				commands.imageBarrier(image, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite, Layout.ColorAttachment, Stage.Transfer,
					Access.TransferRead, Layout.TransferSrc);
				commands.copyImageToBuffer(image, readback);
				commands.bufferBarrier(readback, Stage.Transfer, Access.TransferWrite, Stage.Host, Access.HostRead);
				commands.imageBarrier(image, Stage.Transfer, Access.TransferRead, Layout.TransferSrc, Stage.BottomOfPipe, 0, Layout.Present);
			} else
				commands.imageBarrier(image, Stage.ColorAttachmentOutput, Access.ColorAttachmentWrite, Layout.ColorAttachment, Stage.BottomOfPipe, 0,
					Layout.Present);
			device.submitAndWait(commands);
			if (last) {
				// Bgra8: red is the third byte.
				var pixel = readback.read(0, 4);
				var bgr = [pixel.get(2), pixel.get(1), pixel.get(0)];
				var expected = [for (v in c) Math.round(v * 255)];
				if (bgr.join(",") != expected.join(","))
					throw 'the last frame is ${bgr.join(", ")}, not ${expected.join(", ")}';
				Sys.println('present: ${colors.length} frames to a ${width}x$height window, the last ${expected.join(", ")}');
			}
			swapchain.present();
			commands.close();
			image.close();
		}
		readback.close();
		swapchain.close();
		surface.close();
		device.close();
		instance.close();
		window.close();
	}
}
